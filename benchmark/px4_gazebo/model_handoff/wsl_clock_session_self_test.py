"""Preparation/identity checks only; never UAC, clock changes or flight."""

import ast
from contextlib import ExitStack
from contextlib import redirect_stdout
import hashlib
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from . import wsl_clock_session as session
from .policy_server import validate_policy_seed


class SessionPreparationTests(unittest.TestCase):
    def test_residual_clock_rejection_keeps_original_precondition_and_readback(self):
        clean = dict(offset=0, offset_unit='us', frequency_scaled_ppm=0, tick_us=10000, status=0)
        session.require_cleared_clock(clean)
        for key, value in (('offset', 1908), ('offset', -1), ('frequency_scaled_ppm', 1), ('tick_us', 9999)):
            row = dict(clean, **{key: value})
            with self.assertRaisesRegex(RuntimeError, 'Residual clock adjustment not cleared') as caught:
                session.require_cleared_clock(row)
            self.assertIn(str(value), str(caught.exception))
            self.assertEqual(row[key], value)

    def test_seed_validation_rejects_coercion_and_out_of_range(self):
        for seed in (None, 0, 2**32-1):
            self.assertEqual(validate_policy_seed(seed), seed)
        for seed in (True, -1, 2**32, 0., "0"):
            with self.assertRaises(ValueError):
                validate_policy_seed(seed)

    def test_no_measurement_is_unknown_instead_of_failed_model_score(self):
        for measured in (False, True):
            with TemporaryDirectory() as folder:
                root = Path(folder)
                (root/'admin_result.json').write_text(json.dumps(dict(failure=None, root_held=True,
                    root_done=True, root_resumed_confirmed=True, ubuntu_restore_returncode=0)))
                (root/'ubuntu_result.json').write_text(json.dumps(dict(
                    error=None if measured else 'RuntimeError:Residual clock adjustment not cleared',
                    summary={'quality_passed': True} if measured else None)))
                if measured:
                    path = root/'pilot/condition_00/evaluator/summary.json'
                    path.parent.mkdir(parents=True)
                    path.write_text(json.dumps(dict(category='model_failure', reason='search_local_travel_budget_exceeded')))
                capture = StringIO()
                with patch.object(session.subprocess, 'run', return_value=SimpleNamespace(returncode=0, stdout='123', stderr='')), redirect_stdout(capture):
                    code = session.execute(root, 'pilot')
                report = json.loads(capture.getvalue().splitlines()[-1])
                self.assertTrue(report['restoration_confirmed'])
                self.assertEqual(report['clock_quality_evaluated'], measured)
                self.assertEqual(report['task_evaluated'], measured)
                self.assertEqual(report['clock_quality_passed'], True if measured else None)
                self.assertEqual(report['task_success'], False if measured else None)
                self.assertEqual(code, 3 if measured else 2)

    def fixture(self, stack, folder):
        root = Path(folder)
        repo, data = root / 'repo', root / 'data'
        entry = repo / 'benchmark/px4_gazebo/model_handoff/search_pilot.py'
        entry.parent.mkdir(parents=True)
        entry.write_text('# --role-mode --policy-seed\n', encoding='utf8')
        (data / 'checkpoints').mkdir(parents=True)
        mapping = {}
        for method, (filename, _) in session.PILOT_CHECKPOINTS.items():
            content = method.encode()
            (data / 'checkpoints' / filename).write_bytes(content)
            mapping[method] = filename, hashlib.sha256(content).hexdigest()
        stack.enter_context(patch.object(session, 'os', SimpleNamespace(name='nt')))
        stack.enter_context(patch.object(session, 'drive_path', side_effect=lambda p: (Path(p).as_posix(), '/mnt/test')))
        stack.enter_context(patch.object(session, 'PILOT_CHECKPOINTS', mapping))
        return repo, data, mapping

    def test_all_methods_render_selected_identity_and_unchanged_guardians(self):
        with TemporaryDirectory() as folder, ExitStack() as stack:
            repo, data, mapping = self.fixture(stack, folder)
            guardians = []
            for method, (filename, digest) in mapping.items():
                output = session.prepare(Path(folder)/method, repo, data, 'pilot', method, 7)
                manifest = json.loads((output/'prepared.json').read_text())
                self.assertEqual(manifest['method'], method)
                self.assertEqual(manifest['policy_seed'], 7)
                self.assertEqual(manifest['checkpoint_sha256'], digest)
                source = (output/'run.py').read_text(encoding='utf8')
                self.assertIn("'--method', '"+method+"'", source)
                self.assertIn(filename, source)
                self.assertIn("'--policy-seed', '7'", source)
                self.assertIn("'--role-mode', 'unassigned'", source)
                self.assertIn("'--duration', '90'", source)
                self.assertIn('elapsed > 420', source)
                self.assertNotIn('@@', source)
                for name, expected in manifest['templates'].items():
                    raw = (output/name).read_bytes()
                    ast.parse(raw)
                    self.assertEqual(hashlib.sha256(raw).hexdigest(), expected)
                guardians.append((output/'admin_session.py').read_text(encoding='utf8').replace(output.as_posix(), 'OUTPUT'))
            self.assertTrue(all(g==guardians[0] for g in guardians))

    def test_hash_mismatch_rejected_before_creating_output(self):
        with TemporaryDirectory() as folder, ExitStack() as stack:
            repo, data, mapping = self.fixture(stack, folder)
            (data/'checkpoints'/mapping['joint_bc'][0]).write_bytes(b'changed')
            output = Path(folder)/'attempt'
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                session.prepare(output, repo, data, 'pilot')
            self.assertFalse(output.exists())

    def test_old_checkout_without_policy_seed_rejected_before_output(self):
        with TemporaryDirectory() as folder, ExitStack() as stack:
            repo, data, _ = self.fixture(stack, folder)
            entry = repo/'benchmark/px4_gazebo/model_handoff/search_pilot.py'
            entry.write_text('# --role-mode\n', encoding='utf8')
            output = Path(folder)/'attempt'
            with self.assertRaisesRegex(RuntimeError, 'policy-seed'):
                session.prepare(output, repo, data, 'pilot')
            self.assertFalse(output.exists())

    def test_probe_does_not_load_weights_and_old_evidence_not_overwritten(self):
        with TemporaryDirectory() as folder, ExitStack() as stack:
            repo, data, _ = self.fixture(stack, folder)
            output = Path(folder)/'probe'
            session.prepare(output, repo, Path(folder)/'absent', 'probe')
            original = (output/'prepared.json').read_bytes()
            self.assertFalse((output/'run.py').exists())
            with self.assertRaises(FileExistsError):
                session.prepare(output, repo, data, 'probe')
            self.assertEqual((output/'prepared.json').read_bytes(), original)


if __name__ == '__main__':
    unittest.main()
