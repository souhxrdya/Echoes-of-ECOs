"""python -m eco_library: command overview, info, audit of the committed results, forwarding."""
import contextlib
import io
import json
import unittest

from eco_library import __main__ as cli


class EntryPointTests(unittest.TestCase):
    def test_help_and_unknown_command(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(cli.main([]), 0)
        for command in ("audit", "reproduce", "compare", "info"):
            self.assertIn(command, out.getvalue())
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(cli.main(["no-such-command"]), 2)

    def test_info_reports_the_entries_and_environment(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(cli.main(["info"]), 0)
        info = json.loads(out.getvalue())
        self.assertEqual(sum(info["counts"].values()), len(info["entries"]))
        self.assertEqual(set(info["environment"]), {"python", "implementation", "platform", "machine", "numpy",
                                                    "scipy", "h5py"})

    def test_audit_of_the_committed_results_passes(self):
        result = cli.audit()
        self.assertEqual(result["state"], "PASS", json.dumps(result, indent=1)[:2000])
        self.assertEqual(len(result["steps"]["entry_files"]["files"]), 16)

    def test_forwarded_commands_name_existing_modules(self):
        import importlib
        for module in cli.FORWARD.values():
            self.assertTrue(callable(importlib.import_module(f"eco_library.{module}").main))


if __name__ == "__main__":
    unittest.main()
