"""Commands the VR viewer shares read only the Command_Engine names its own Command_Engine defines.

In the VR process opt_vr/src/Command_Engine.py shadows main's module and re-exports main's names one
by one, with no fallback: a shared command that reads a new main-side name (Command_Engine.NEW_NAME)
fails there, at import for a module-level read and at run time otherwise. A constant that commands
share belongs in a module VR does not shadow (desktop/Node_Colors.py).

The check needs the opt_vr checkout next to src (or EMAPSSN_VR_ROOT pointing at it) and is skipped
without one.
"""

import ast
import os
import unittest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VR_ROOT = os.environ.get("EMAPSSN_VR_ROOT") or os.path.join(PROJECT_ROOT, "opt_vr")
VR_ENGINE = os.path.join(VR_ROOT, "src", "Command_Engine.py")
VR_COMMANDS = os.path.join(VR_ROOT, "src", "commands")
COMMANDS = os.path.join(PROJECT_ROOT, "src", "commands")


def names_defined(path):
    """The names a module binds at its top level."""
    with open(path, encoding="utf-8") as handle:
        tree = ast.parse(handle.read())
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update((alias.asname or alias.name).split(".")[0] for alias in node.names)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            names.add(node.id)
    return names


def engine_names_read(path):
    """{name: line} of every ``Command_Engine.<name>`` a module reads."""
    with open(path, encoding="utf-8") as handle:
        tree = ast.parse(handle.read())
    reads = {}
    for node in ast.walk(tree):
        if (isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load)
                and isinstance(node.value, ast.Name) and node.value.id == "Command_Engine"):
            reads.setdefault(node.attr, node.lineno)
    return reads


class SharedCommandsUseVrEngineNamesTests(unittest.TestCase):
    @unittest.skipUnless(os.path.isfile(VR_ENGINE), "no opt_vr checkout to compare with")
    def test_every_engine_name_a_shared_command_reads_exists_in_the_vr_engine(self):
        defined = names_defined(VR_ENGINE)
        own = set(os.listdir(VR_COMMANDS)) if os.path.isdir(VR_COMMANDS) else set()
        missing = []
        for name in sorted(os.listdir(COMMANDS)):
            if not name.endswith(".py") or name == "__init__.py" or name in own:
                continue
            for attribute, line in engine_names_read(os.path.join(COMMANDS, name)).items():
                if attribute not in defined:
                    missing.append(f"{name}:{line} Command_Engine.{attribute}")
        self.assertEqual(missing, [], "opt_vr's Command_Engine lacks these names")

    def test_the_name_collector_sees_reads_and_definitions(self):
        import tempfile
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "sample.py")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write("import Command_Engine\nX = Command_Engine.NOISE_COLOR\nCommand_Engine.show_status(1)\n"
                             "def helper():\n    pass\n")
            self.assertEqual(set(engine_names_read(path)), {"NOISE_COLOR", "show_status"})
            self.assertTrue({"X", "helper", "Command_Engine"} <= names_defined(path))


if __name__ == "__main__":
    unittest.main()
