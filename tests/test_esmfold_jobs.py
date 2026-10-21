# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""The Fold View page while structures are being folded (esmfold_jobs.js).

The Viewer's folding worker reports a run to the page (esmfold_job events: started,
progress, structure_failed, failed, finished) and the Viewer says on connect whether
it has a window (viewer_info). The page shows a row for the sequence being folded, a
progress bar on the canvas while there is no structure yet, a row for each failure
with its reason until it is dismissed, and, for a Viewer with no window, a badge and
an explanation. The tests give the page's event stream those events, as the backend
broadcasts them (tests/test_esmfold_backend.py covers the backend), and read its DOM.
"""

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for folder in (SRC, ROOT):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

from tests.test_esmfold_colors import A, B
from tests.test_esmfold_structures import StructuresTestCase


class JobTestCase(StructuresTestCase):
    def event(self, event, **fields):
        self.broadcast({"type": "esmfold_job", "event": event, **fields})
        self.settle(300)

    def rows(self):
        return self.script("""
            return [...document.querySelectorAll('#struct-list .srow')].map(row => ({
                classes: row.className, name: (row.querySelector('.srow-name span:not(.chip)') || row.querySelector('.srow-name')).textContent,
                sub: (row.querySelector('.srow-sub') || {}).textContent, chips: [...row.querySelectorAll('.chip')].map(c => c.textContent),
                buttons: [...row.querySelectorAll('button')].map(b => b.textContent), spinner: !!row.querySelector('.icon-spin')}));
            """)

    def tile(self):
        return self.script("""
            const bar = document.getElementById('folding-bar');
            return {title: document.getElementById('folding-title').textContent, name: document.getElementById('folding-name').textContent,
                    now: bar.getAttribute('aria-valuenow'), max: bar.getAttribute('aria-valuemax'), width: bar.firstElementChild.style.width,
                    elapsed: document.getElementById('folding-elapsed').textContent};
            """)

    def view_with_a_canvas(self):
        """The view the page is in if there were a canvas (this engine has none): the rule, with one stood in."""
        return self.script("""
            const own = Object.getOwnPropertyDescriptor(plugin, 'canvas3d');
            Object.defineProperty(plugin, 'canvas3d', {value: {}, configurable: true, writable: true});
            try { updateView(); return document.body.dataset.view; }
            finally { if (own) Object.defineProperty(plugin, 'canvas3d', own); else delete plugin.canvas3d; updateView(); }
            """)


class ProgressTests(JobTestCase):
    def test_a_run_shows_the_sequence_being_folded_and_a_bar_until_the_first_structure_arrives(self):
        self.write_all()
        self.open_page()
        self.assertEqual(self.view_with_a_canvas(), "empty")
        self.event("started", total=3, names=["node_a", "node_b", "node_c"])
        waiting = self.tile()
        self.assertEqual(waiting["title"], "Getting the model ready...")
        self.assertEqual(self.view_with_a_canvas(), "folding")
        pending = self.rows()
        self.assertEqual(len(pending), 1)
        self.assertIn("pending", pending[0]["classes"])
        self.assertTrue(pending[0]["spinner"])
        self.assertEqual(pending[0]["name"], "Getting the model ready...")
        self.event("progress", index=2, total=3, name="node_b")
        tile = self.tile()
        self.assertEqual((tile["title"], tile["name"], tile["now"], tile["max"]), ("Folding 2 of 3", "node_b", "1", "3"))
        self.assertTrue(tile["width"].startswith("33.3"), tile["width"])
        row = self.rows()[0]
        self.assertEqual((row["name"], row["sub"][:len("Folding 2 of 3")]), ("node_b", "Folding 2 of 3"))
        # The clock counts the time since the run started.
        self.script("job.startedAt = Date.now() - 65000; tickJob(); return 1")
        self.assertRegex(self.script("return document.querySelector('.job-elapsed').textContent"), r"^01:0[5-9]$")
        # A structure arrives: the canvas has it, the card lists it, and the row for the next one stays.
        self.push(A)
        arrived = self.rows()
        self.assertEqual([("pending" in row["classes"]) for row in arrived], [False, True])
        self.assertEqual(self.view_with_a_canvas(), "ready")
        self.event("finished", succeeded=3, total=3, failed=False)
        self.assertEqual([("pending" in row["classes"]) for row in self.rows()], [False])
        self.assertEqual(self.script("return document.getElementById('toast').querySelector('span').textContent"), "Folded 3 of 3.")
        self.assertEqual(self.view_with_a_canvas(), "ready")
        self.assertEqual(self.console_errors(), [])

    def test_an_older_viewer_that_sends_no_events_changes_nothing(self):
        self.write_all()
        self.open_page()
        self.push(A)
        self.assertEqual([row["name"] for row in self.rows()], ["11"])
        self.assertIsNone(self.script("return job"))

    def test_a_run_that_ends_without_a_word_of_finished_is_not_a_pending_row_once_it_failed(self):
        self.write_all()
        self.open_page()
        self.event("started", total=2, names=["node_a", "node_b"])
        self.event("progress", index=1, total=2, name="node_a")
        self.event("failed", reason="the model could not be loaded", log=r"C:\runs\worker.stdout")
        rows = self.rows()
        self.assertEqual(len(rows), 1)
        self.assertIn("failed", rows[0]["classes"])
        self.assertEqual((rows[0]["name"], rows[0]["chips"]), ("node_a", ["Failed"]))
        self.assertEqual(rows[0]["sub"], "the model could not be loaded")
        self.assertEqual(self.script("return jobRunning()"), False)


class FailureTests(JobTestCase):
    def test_a_structure_that_failed_has_a_row_with_its_reason_until_it_is_dismissed(self):
        self.write_all()
        self.open_page()
        self.push(A)
        self.event("structure_failed", name="node_b", reason="CUDA out of memory")
        rows = self.rows()
        self.assertEqual([row["name"] for row in rows], ["11", "node_b"])
        failed = rows[1]
        self.assertIn("failed", failed["classes"])
        self.assertEqual((failed["chips"], failed["sub"], failed["buttons"]), (["Failed"], "CUDA out of memory", ["Dismiss"]))
        toast = self.script("const t = document.getElementById('toast'); return [t.classList.contains('error'), t.querySelector('span').textContent]")
        self.assertEqual(toast, [True, "Folding node_b failed: CUDA out of memory"])
        # The structure that did fold is still usable beside it.
        self.assertEqual(self.script("return loadedStructures().length"), 1)
        self.click("#struct-list .srow.failed button", 400)
        self.assertEqual([row["name"] for row in self.rows()], ["11"])

    def test_a_failure_with_a_log_can_copy_its_path(self):
        self.write_all()
        self.open_page()
        self.script("""
            window.__copied = [];
            Object.defineProperty(navigator, 'clipboard', {value: {writeText: text => { window.__copied.push(text); return Promise.resolve(); }}, configurable: true});
            return 1;
            """)
        self.event("structure_failed", name="node_b", reason="stopped", log=r"C:\runs\worker.stdout")
        self.assertEqual(self.rows()[0]["buttons"], ["Copy log path", "Dismiss"])
        self.click("#struct-list .srow.failed .btn", 300)
        self.assertEqual(self.script("return window.__copied"), [r"C:\runs\worker.stdout"])
        self.assertEqual(self.script("return document.getElementById('toast').querySelector('span').textContent"), "Copied the log path.")

    def test_a_structure_that_cannot_be_loaded_has_a_row_too(self):
        self.write_all()
        self.open_page()
        self.broadcast({"type": "esmfold_pdb", "node_id": "missing", "pdb_url": "/structures/missing.pdb"})
        self.settle(1500)
        rows = self.rows()
        self.assertEqual(len(rows), 1)
        self.assertEqual((rows[0]["name"], rows[0]["chips"]), ("missing", ["Failed"]))
        self.assertEqual(rows[0]["sub"], "Mol* could not get a structure from it.")
        # (It said it loaded it, as the pivot, before: nothing was loaded.)
        self.assertEqual(self.script("return document.getElementById('toast').querySelector('span').textContent"), "Failed to load structure missing.")
        self.assertEqual(self.script("return loadedStructures().length"), 0)
        self.click("#struct-list .srow.failed button", 300)
        self.assertEqual(self.rows(), [])

    def test_a_file_that_is_not_a_structure_has_a_failure_row_and_no_success_message(self):
        self.write_all()
        self.open_page()
        self.script("openStructureFiles([new File(['this is not a pdb file'], 'bad.pdb')]); await structureQueue; return 1")
        self.settle(800)
        rows = self.rows()
        self.assertEqual([(row["name"], row["chips"]) for row in rows], [("bad.pdb", ["Failed"])])
        self.assertEqual(self.script("return loadedStructures().length"), 0)
        self.assertEqual(self.script("return document.getElementById('toast').querySelector('span').textContent"), "Failed to load structure bad.pdb.")

    def test_the_empty_state_gives_way_to_a_failure_row_not_to_both(self):
        self.write_all()
        self.open_page()
        self.assertEqual(self.script("return document.querySelector('#struct-list .hint').textContent"), "Structures appear here as they are folded.")
        self.event("structure_failed", name="node_b", reason="stopped")
        self.assertEqual(self.script("return document.querySelectorAll('#struct-list .hint').length"), 0)
        self.click("#struct-list .srow.failed button", 300)
        self.assertEqual(self.script("return document.querySelector('#struct-list .hint').textContent"), "Structures appear here as they are folded.")


class NoWindowTests(JobTestCase):
    def test_a_viewer_with_no_window_gets_a_badge_and_an_explanation(self):
        self.write_all()
        self.open_page()
        shown = "return {badge: !document.getElementById('hidden-badge').hidden, hidden: document.body.dataset.hidden, text: getComputedStyle(document.querySelector('.nowindow')).display}"
        self.assertEqual(self.script(shown), {"badge": False, "text": "none"})
        self.broadcast({"type": "viewer_info", "hidden": True})
        self.settle(300)
        self.assertEqual(self.script(shown), {"badge": True, "hidden": "1", "text": "block"})
        self.assertEqual(self.script("return document.getElementById('hidden-badge').textContent"), "No window")
        self.broadcast({"type": "viewer_info", "hidden": False})
        self.settle(300)
        self.assertEqual(self.script(shown), {"badge": False, "hidden": "0", "text": "none"})
        self.assertEqual(self.console_errors(), [])


if __name__ == "__main__":
    unittest.main()
