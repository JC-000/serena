"""
CA65 language server integration for Serena.

Spawns the bundled `ca65-ls` (Python + pygls + tree-sitter-ca65) as a subprocess
over stdio. ca65-ls itself lives in a sibling package:
`github.com/JC-000/ca65-asm-serena-lsp` under `packages/ca65-ls/`.

Modeled on PyrightServer (also a Python-based, pip-installable LSP).
"""

import hashlib
import importlib.util
import logging
import os
import shutil
import sys
from collections.abc import Hashable
from pathlib import Path

from overrides import override

from solidlsp.ls import LanguageServerDependencyProvider, LanguageServerDependencyProviderSinglePath, SolidLanguageServer
from solidlsp.ls_config import LanguageServerConfig
from solidlsp.ls_exceptions import SolidLSPException
from solidlsp.settings import SolidLSPSettings

log = logging.getLogger(__name__)


class Ca65LanguageServer(SolidLanguageServer):
    """
    CA65 (6502/65C02/65816 assembly) language server.

    ca65-ls runs as a Python module: `python -m ca65_ls.server --stdio`. Users
    install the `ca65-ls` package alongside Serena; pyproject.toml's optional
    extra `[ca65]` pulls it in.
    """

    DEFAULT_INITIALIZE_TIMEOUT = 60.0
    """
    Seconds to wait for ca65-ls to answer `initialize` (it indexes the whole project synchronously
    inside that request; a cold index of a few thousand files takes a couple of seconds). Overridable
    through the LS-specific setting `initialize_timeout`. Bounded separately from the generic request
    timeout so that a ca65-ls that hangs at startup fails in seconds, not minutes.
    """

    INSTALL_HINT = (
        "Install ca65-ls into the Python environment Serena runs under ({python}): "
        "`{python} -m pip install ca65-ls`, or from a source checkout of ca65-asm-serena-lsp "
        "`uv pip install -e ~/Documents/ca65-asm-serena-lsp/packages/ca65-ls` into Serena's venv."
    )

    def __init__(self, config: LanguageServerConfig, repository_root_path: str, solidlsp_settings: SolidLSPSettings):
        self._configured_request_timeout: float | None = None
        # ca65-ls reports resolved (symlink-free) file URIs, and SolidLSP derives relative paths by comparing
        # resolved locations against this root as given. A project opened through a symlink would therefore
        # get `../../..` relative paths and `request_full_symbol_tree` would raise ValueError, so resolve the
        # root up front. (Serena's Project already resolves it; this covers direct SolidLSP use.)
        super().__init__(
            config,
            os.path.realpath(repository_root_path),
            None,
            "ca65",
            solidlsp_settings,
        )

    def _create_dependency_provider(self) -> LanguageServerDependencyProvider:
        return self.DependencyProvider(self._custom_settings, self._ls_resources_dir)

    class DependencyProvider(LanguageServerDependencyProviderSinglePath):
        def _get_or_install_core_dependency(self) -> str:
            # Use the same interpreter Serena itself is running under. The user
            # must have installed ca65-ls into that environment (either via
            # Serena's `[ca65]` extra, or with `uv pip install ca65-ls`).
            return sys.executable

        def _create_launch_command(self, core_path: str) -> list[str]:
            return [core_path, "-m", "ca65_ls.server", "--stdio"]

    @override
    def is_ignored_dirname(self, dirname: str) -> bool:
        return super().is_ignored_dirname(dirname) or dirname in (
            "build",
            "obj",
            ".ca65-ls",  # ca65-ls cache directory
            ".claude",  # Claude Code state; `.claude/worktrees/agent-*/` holds full copies of the project
        )

    @override
    def _get_wait_time_for_cross_file_referencing(self) -> float:
        # ca65-ls builds its whole workspace index synchronously inside `initialize`, so cross-file
        # references are complete as soon as the server has started; the generic 2 s grace period
        # would only delay the first definition/references request of every server start.
        return 0.0

    @staticmethod
    @override
    def _determine_log_level(line: str) -> int:
        # pygls (ca65-ls's LSP framework) logs its protocol traffic through the standard `logging`
        # module, which ends up on stderr as `INFO:pygls.protocol...:Sending data: {...}` / `Received ...`.
        # Such a line is chatter, not an error, even when the payload happens to mention a symbol like
        # `ip65_error`; the default classifier would report it at ERROR.
        level_name, _, rest = line.lstrip().partition(":")
        if level_name in ("DEBUG", "INFO") and rest.startswith("pygls"):
            return logging.DEBUG
        return SolidLanguageServer._determine_log_level(line)

    @override
    def set_request_timeout(self, timeout: float | None) -> None:
        self._configured_request_timeout = timeout
        super().set_request_timeout(timeout)

    def _initialize_timeout(self) -> float:
        timeout = self.custom_settings.get("initialize_timeout", self.DEFAULT_INITIALIZE_TIMEOUT)
        try:
            timeout = float(timeout)
        except (TypeError, ValueError):
            log.warning("Ignoring invalid 'initialize_timeout' setting %r; using %s s", timeout, self.DEFAULT_INITIALIZE_TIMEOUT)
            timeout = self.DEFAULT_INITIALIZE_TIMEOUT
        if self._configured_request_timeout is not None:
            timeout = min(timeout, self._configured_request_timeout)
        return timeout

    def _check_launch_command(self, cmd: list[str]) -> None:
        """
        Fails fast, with an actionable message, when the launch command cannot possibly start ca65-ls.
        Without this, a missing install surfaces as a generic initialize failure whose cause
        (`No module named ca65_ls`) is only visible on a separate stderr log line.
        """
        hint = self.INSTALL_HINT.format(python=sys.executable)
        if not cmd:
            raise RuntimeError(f"ca65-ls launch command is empty. {hint}")
        executable = cmd[0]
        if shutil.which(executable) is None:
            raise RuntimeError(f"ca65-ls launcher not found or not executable: {executable!r} (launch command: {cmd}). {hint}")
        runs_module_in_own_interpreter = (
            "-m" in cmd and "ca65_ls.server" in cmd and os.path.realpath(executable) == os.path.realpath(sys.executable)
        )
        if runs_module_in_own_interpreter and importlib.util.find_spec("ca65_ls") is None:
            raise RuntimeError(f"ca65-ls is not installed (module `ca65_ls` not importable by {sys.executable}). {hint}")

    def _create_base_initialize_params(self) -> dict:
        """Return the CA65-specific InitializeParams basis.

        ``processId``, ``rootPath``, ``rootUri``, ``clientInfo`` and
        ``workspaceFolders`` are populated by the default
        ``InitializeParamsBuilder`` and must not be set here. ``initializationOptions``
        is ca65-ls-specific (``projectRoot`` in particular is not supplied by the
        builder) and is preserved by the builder, which merges any user-provided
        ``ls_specific_settings`` initialization options on top of it.
        """
        return {
            "initializationOptions": {
                "projectRoot": self.repository_root_path,
                "buildDirs": ["build", "obj"],
                "cpu": "6502",
            },
            "capabilities": {
                "workspace": {
                    "workspaceEdit": {"documentChanges": True},
                    "symbol": {
                        "dynamicRegistration": True,
                        "symbolKind": {"valueSet": list(range(1, 27))},
                    },
                },
                "textDocument": {
                    "synchronization": {"didSave": True},
                    "definition": {"dynamicRegistration": True},
                    "references": {"dynamicRegistration": True},
                    "documentSymbol": {
                        "dynamicRegistration": True,
                        "symbolKind": {"valueSet": list(range(1, 27))},
                        "hierarchicalDocumentSymbolSupport": True,
                    },
                    "publishDiagnostics": {"relatedInformation": True},
                },
            },
        }

    @override
    def _raw_document_symbols_cache_fingerprint(self) -> Hashable | None:
        """Auto-invalidate Serena's on-disk document-symbol cache whenever the
        ca65-ls modules that shape its output change.

        This is the *raw* fingerprint hook: a change to ca65-ls alters the
        symbol payload the language server itself returns (not Serena's
        high-level post-processing), so the raw cache — and, transitively, the
        high-level cache built on top of it — must be invalidated.

        Without this, Serena trusts its `.serena/cache/ca65/*.pkl` files across
        process restarts.  In development (with `--with-editable` installs)
        that means symbol output produced by an old ca65-ls keeps getting
        served even after a code update.  The fix is to include in the cache
        fingerprint:

          1. ca65-ls's installed package version (covers pinned releases), and
          2. a sha256 over the source files whose behavior directly affects
             document-symbol output (covers editable-install development).

        Any change to buffer/document.py, types.py, or server.py flips the
        hash, which flips this fingerprint, which causes Serena's cache to
        miss and rebuild from the live LSP.
        """
        try:
            import ca65_ls  # local import: avoid hard-failing if the user hasn't installed it yet
        except ImportError:
            return None

        version = getattr(ca65_ls, "__version__", "unknown")
        ca65_ls_root = Path(ca65_ls.__file__).parent
        source_files = [
            ca65_ls_root / "buffer" / "document.py",
            ca65_ls_root / "types.py",
            ca65_ls_root / "server.py",
        ]
        h = hashlib.sha256()
        for f in source_files:
            try:
                h.update(f.read_bytes())
            except OSError:
                # File missing (e.g. partial install); fold that into the hash
                # so the fingerprint still varies between "missing" and "present".
                h.update(b"<missing:" + str(f).encode() + b">")
        return (version, h.hexdigest()[:16])

    def _start_server(self) -> None:
        def do_nothing(_params: dict) -> None:
            return

        self.server.on_request("client/registerCapability", do_nothing)
        self.server.on_notification("window/logMessage", do_nothing)
        self.server.on_notification("textDocument/publishDiagnostics", do_nothing)

        cmd = list(self._get_process_launch_info().cmd)
        self._check_launch_command(cmd)

        log.info("Starting ca65-ls server process")
        self.server.start()

        params = self._create_initialize_params()
        initialize_timeout = self._initialize_timeout()
        self.server.set_request_timeout(initialize_timeout)
        try:
            init_response = self.server.send.initialize(params)
        except TimeoutError as e:
            raise RuntimeError(
                f"ca65-ls did not answer `initialize` within {initialize_timeout:g} s (launch command: {cmd}). "
                "Raise the LS-specific setting `initialize_timeout` if the project is very large; "
                "otherwise the server is hanging at startup — run the command by hand to see why."
            ) from e
        except SolidLSPException as e:
            if not e.is_language_server_terminated():
                raise
            raise RuntimeError(
                f"ca65-ls exited during `initialize` (launch command: {cmd}); its stderr was logged just above. "
                + self.INSTALL_HINT.format(python=sys.executable)
            ) from e
        finally:
            self.server.set_request_timeout(self._configured_request_timeout)
        log.debug("ca65-ls initialize response: %s", init_response)

        assert "textDocumentSync" in init_response["capabilities"]
        assert "documentSymbolProvider" in init_response["capabilities"]
        assert "definitionProvider" in init_response["capabilities"]
        assert "referencesProvider" in init_response["capabilities"]
        assert "workspaceSymbolProvider" in init_response["capabilities"]
        assert "hoverProvider" in init_response["capabilities"]
        assert "renameProvider" in init_response["capabilities"]

        self.server.notify.initialized({})
