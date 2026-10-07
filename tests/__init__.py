"""EMAP-SSN regression tests.

Importing this package makes a test run independent of the developer's own
files and shell, whichever subset of modules runs:

* SSN_* variables inherited from the shell are dropped, so a stray
  SSN_VIEWER_SETTINGS_PATH or SSN_TOOL_SETTINGS_FILE cannot steer a test.
* SSN_VIEWER_EXPLICIT_SETTINGS=1 stops EMAPSSN_Config from applying the
  developer's viewer_settings.json when it is imported (the Viewer sets the same
  flag when it starts from explicit settings).
* SSN_VIEWER_SESSION_DIR points at a folder private to this run, so web servers
  started by tests never publish Viewer session descriptors (with live tokens)
  where real MCP clients look for Viewers. The folder is removed at exit.
* QT_QPA_PLATFORM defaults to offscreen, so no Qt window appears.
* Every script in src/tools applies <project>/tools_settings.json to its module
  globals when it is imported. An import hook points each tool import at a
  missing settings file instead, unless the test already chose a settings file
  for that script (SSN_TOOL_SETTINGS_SCRIPT and SSN_TOOL_SETTINGS_FILE).
  Without the hook, whichever test module imported a tool first decided the
  globals that every later module saw.

Processes that tests spawn inherit the environment of the first test process,
so the environment is only reset there; the import hook is installed in every
process that imports this package. Pool workers that a tool starts with the
spawn method import the tool without importing this package, so they apply
<project>/tools_settings.json unless the test set SSN_TOOL_SETTINGS_SCRIPT and
SSN_TOOL_SETTINGS_FILE for that tool, as a tool's main() does for its workers.
Pass anything a worker's result depends on explicitly, or set those variables.
"""
import atexit
import importlib.abc
import importlib.machinery
import os
import shutil
import sys
import tempfile

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TOOLS_DIR = os.path.join(_PROJECT_ROOT, "src", "tools")
MISSING_TOOL_SETTINGS = os.path.join(_PROJECT_ROOT, "tests", "nonexistent-settings.json")
_ROOT_PROCESS_MARKER = "EMAPSSN_TEST_ENVIRONMENT"

if not os.environ.get(_ROOT_PROCESS_MARKER):
    os.environ[_ROOT_PROCESS_MARKER] = str(os.getpid())
    for _name in [name for name in os.environ if name.startswith("SSN_")]:
        del os.environ[_name]
    os.environ["SSN_VIEWER_EXPLICIT_SETTINGS"] = "1"
    _SESSION_DIRECTORY = tempfile.mkdtemp(prefix="emapssn-test-sessions-")
    os.environ["SSN_VIEWER_SESSION_DIR"] = _SESSION_DIRECTORY
    atexit.register(shutil.rmtree, _SESSION_DIRECTORY, ignore_errors=True)
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


class _ToolSettingsLoader(importlib.abc.Loader):
    """Execute a tool script's module code with its settings file pointed elsewhere."""

    def __init__(self, loader, script_name):
        self._loader = loader
        self._script_name = script_name

    def create_module(self, spec):
        return self._loader.create_module(spec)

    def exec_module(self, module):
        keys = ("SSN_TOOL_SETTINGS_SCRIPT", "SSN_TOOL_SETTINGS_FILE")
        saved = {key: os.environ.get(key) for key in keys}
        chosen = saved["SSN_TOOL_SETTINGS_SCRIPT"] == self._script_name and saved["SSN_TOOL_SETTINGS_FILE"]
        if not chosen:
            os.environ["SSN_TOOL_SETTINGS_SCRIPT"] = self._script_name
            os.environ["SSN_TOOL_SETTINGS_FILE"] = MISSING_TOOL_SETTINGS
        try:
            self._loader.exec_module(module)
        finally:
            for key, value in saved.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    def __getattr__(self, name):
        # get_source, get_filename, ... for tracebacks, inspect and numba caching.
        return getattr(self._loader, name)


class _ToolSettingsFinder(importlib.abc.MetaPathFinder):
    """Wrap the loader of src/tools/<name>.py, imported as <name> or tools.<name>."""

    def __init__(self):
        try:
            names = os.listdir(_TOOLS_DIR)
        except OSError:
            names = []
        self._scripts = {
            name[:-3] for name in names if name.endswith(".py") and not name.startswith("_")
        }

    def find_spec(self, fullname, path=None, target=None):
        package, _, leaf = fullname.rpartition(".")
        if leaf not in self._scripts or package not in ("", "tools"):
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        if spec is None or spec.origin is None or spec.loader is None:
            return None
        expected = os.path.join(_TOOLS_DIR, leaf + ".py")
        if os.path.normcase(os.path.abspath(spec.origin)) != os.path.normcase(expected):
            return None
        spec.loader = _ToolSettingsLoader(spec.loader, leaf + ".py")
        return spec


if not any(isinstance(finder, _ToolSettingsFinder) for finder in sys.meta_path):
    sys.meta_path.insert(0, _ToolSettingsFinder())
