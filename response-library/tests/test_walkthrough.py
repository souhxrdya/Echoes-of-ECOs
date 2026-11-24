"""The walkthrough notebook: its stored outputs are complete and free of errors, and every code cell still
runs against the committed results."""
import contextlib
import io
import json
import os
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO/"walkthrough.ipynb"


class StoredNotebook(unittest.TestCase):
    def test_every_code_cell_has_stored_output_and_no_error(self):
        nb = json.loads(NOTEBOOK.read_text(encoding="utf8"))
        code = [c for c in nb["cells"] if c["cell_type"] == "code"]
        self.assertGreater(len(code), 10)
        for c in code:
            kinds = {o["output_type"] for o in c["outputs"]}
            self.assertNotIn("error", kinds)
        self.assertTrue(all(c["outputs"] for c in code[:1] + code[2:]))    # the style cell prints nothing
        text = NOTEBOOK.read_text(encoding="utf8")
        self.assertNotIn("/sessions/", text)
        self.assertNotIn("C:\\\\Users", text)


class RunsAgainstTheLibrary(unittest.TestCase):
    def test_all_code_cells_execute(self):
        nb = json.loads(NOTEBOOK.read_text(encoding="utf8"))
        cwd, env = os.getcwd(), dict(os.environ)
        os.environ.update(ECO_LIBRARY_ROOT=str(REPO), MPLBACKEND="Agg")
        namespace = {"__name__": "__main__"}
        try:
            with contextlib.redirect_stdout(io.StringIO()) as out:
                for i, cell in enumerate(nb["cells"]):
                    if cell["cell_type"] == "code":
                        exec(compile("".join(cell["source"]), f"walkthrough cell {i}", "exec"), namespace)
        finally:
            os.chdir(cwd)
            os.environ.clear()
            os.environ.update(env)
        printed = out.getvalue()
        # bitwise on the computer type that produced the results; elsewhere a roundoff-level difference is stated
        self.assertTrue("BITWISE_IDENTICAL" in printed or "DIFFERS (expected only on another kind" in printed)
        self.assertIn("16 of 16 result files match the index and pass their audit", printed)
        self.assertIn("status audit of the catalogue: no contradiction", printed)


if __name__ == "__main__":
    unittest.main()
