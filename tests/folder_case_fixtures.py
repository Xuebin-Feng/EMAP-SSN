"""Folders that ignore letter case, or keep it apart, for the tests of
utilities.Output_Names.folder_ignores_case and the commands that use it.

ignores_case() is the ground truth, found by writing two files.
make_case_sensitive() turns an empty folder case-sensitive, or skips the test.
"""
import os
import shutil
import subprocess
import tempfile


def ignores_case(folder):
    """The ground truth for `folder`: write case-twin and CASE-TWIN in a new
    subfolder, which takes the folder's case setting, and count the files."""
    twins = tempfile.mkdtemp(dir=folder)
    try:
        for name in ("case-twin", "CASE-TWIN"):
            with open(os.path.join(twins, name), "w", encoding="utf-8") as handle:
                handle.write(name)
        return len(os.listdir(twins)) == 1
    finally:
        shutil.rmtree(twins)


def make_case_sensitive(test, folder):
    """Make the empty `folder` keep letter case apart, or skip `test`.

    Linux folders usually do already. On Windows, fsutil sets a folder's case
    sensitivity when the WSL feature is installed; new subfolders inherit it.
    """
    if os.name == "nt":
        try:
            subprocess.run(
                ["fsutil.exe", "file", "setCaseSensitiveInfo", folder, "enable"],
                capture_output=True, check=False, timeout=60,
            )
        except (OSError, subprocess.TimeoutExpired):
            pass
    if ignores_case(folder):
        test.skipTest("No case-sensitive folder can be made here.")
