"""Prepare/replay the locally verified, bounded WSL clock isolation session.

Preparation only by default. --execute requests UAC and waits for restoration.
The pilot is one frozen model diagnostic, never a training or transfer score.
"""

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time


TEMPLATES = Path(__file__).with_name('wsl_clock_templates')
PILOT_CHECKPOINTS = {
    'joint_bc': ('joint_bc_confirm_fix_v1_seed_0.pt',
                 'd65ba56323d2a1e426e2c810bbaccf5e33f9a0a7c36201b1d37b87ab197f93af'),
    'dual_vln_hier': ('dual_vln_hier_confirm_fix_v1_seed_0.pt',
                      'e6cc49b672f2c87453147d4931c21bfd3b8deb1ecf0ff1ede25698a409ada8e5'),
    'dual_wam_mpc': ('dual_wam_mpc_confirm_fix_v1_seed_0.pt',
                     '53119eae2f847b1b7b48036b608584a89fb511129c22abdefd9a50dd093780e6'),
}


def require_cleared_clock(readback):
    """Preserve the zero-residual precondition, with actionable diagnostics."""
    observed = {key: readback[key] for key in ('offset', 'offset_unit', 'frequency_scaled_ppm', 'tick_us', 'status')}
    if (readback['offset'] != 0 or readback['frequency_scaled_ppm'] != 0
            or readback['tick_us'] != 10000):
        raise RuntimeError('Residual clock adjustment not cleared: ' + json.dumps(observed, sort_keys=True))


def drive_path(path):
    value = Path(path).resolve().as_posix()
    # These paths are embedded in reviewed Python and VM-shell templates.
    # Refuse shell/Python interpolation characters rather than guessing escapes.
    if not re.fullmatch(r'[A-Za-z]:/[A-Za-z0-9_ ./:-]+', value):
        raise ValueError('Use an absolute Windows drive path with simple characters')
    return value, '/mnt/' + value[0].lower() + value[2:]


def prepare(output, repo, data, mode, method='joint_bc', policy_seed=0):
    from .policy_server import validate_policy_seed
    validate_policy_seed(policy_seed)
    if policy_seed is None:
        raise ValueError('Clock-isolated pilots require an explicit policy seed')
    if mode not in ('probe', 'pilot') or method not in PILOT_CHECKPOINTS:
        raise ValueError('Unsupported session mode/method')
    if os.name != 'nt':
        raise RuntimeError('Launch this entry from Windows, not inside WSL')
    windows_output, wsl_output = drive_path(output)
    _, wsl_repo = drive_path(repo)
    _, wsl_data = drive_path(data)
    if mode == 'pilot':
        entry = Path(repo) / 'benchmark/px4_gazebo/model_handoff/search_pilot.py'
        capabilities = entry.read_text(encoding='utf-8')
        if any(flag not in capabilities for flag in ('--role-mode', '--policy-seed')):
            raise RuntimeError('Pilot requires unassigned-role and policy-seed search wiring; use probe on older checkouts')
    filename, expected = PILOT_CHECKPOINTS[method]
    if mode == 'pilot':
        actual = hashlib.sha256((Path(data) / 'checkpoints' / filename).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError('Frozen checkpoint hash mismatch; no session prepared')
    replacements = {'@@WINDOWS_OUTPUT@@': windows_output, '@@WSL_OUTPUT@@': wsl_output,
                    '@@WSL_REPO@@': wsl_repo, '@@WSL_DATA@@': wsl_data,
                    '@@METHOD@@': method, '@@CHECKPOINT_FILE@@': filename,
                    '@@CHECKPOINT_SHA256@@': expected, '@@POLICY_SEED@@': str(policy_seed)}
    selected = {'admin_session.py': 'admin_session.py.in',
                'ubuntu_ready.py': 'ubuntu_ready.py.in',
                'ubuntu_action.py': 'probe_action.py.in' if mode == 'probe' else 'ubuntu_action.py.in'}
    if mode == 'pilot':
        selected['run.py'] = 'run.py.in'
    rendered = {}
    for target, template in selected.items():
        source = (TEMPLATES / template).read_text(encoding='utf-8')
        for key, value in replacements.items():
            source = source.replace(key, value)
        if '@@' in source:
            raise ValueError('Unresolved session template token')
        ast.parse(source, filename=target)
        rendered[target] = source
    root = Path(windows_output)
    root.mkdir(parents=True, exist_ok=False)  # Never overwrite a previous attempt.
    manifest = {'schema': 'aerowam-wsl-clock-session/1-draft', 'mode': mode,
                'executed': False, 'distro': 'Ubuntu-22.04', 'model_user': 'ryx',
                'permanent_config_change': False, 'clock_limit_ms': 10,
                'inference_budget_ms': 400, 'training': False, 'templates': {}}
    if mode == 'pilot':
        manifest.update(method=method, policy_seed=policy_seed, checkpoint_sha256=expected)
    for name, source in rendered.items():
        raw = source.encode('utf-8')
        with (root / name).open('xb') as stream:
            stream.write(raw)
        manifest['templates'][name] = hashlib.sha256(raw).hexdigest()
    with (root / 'prepared.json').open('x', encoding='utf-8') as stream:
        json.dump(manifest, stream, indent=2)
    return root


def execute(root, mode):
    def quote(value):
        return "'" + str(value).replace("'", "''") + "'"
    arguments = subprocess.list2cmdline(['-X', 'utf8', '-B', str(root / 'admin_session.py')])
    script = ('Start-Process -FilePath ' + quote(sys.executable) + ' -ArgumentList '
              + quote(arguments) + ' -Verb RunAs -WindowStyle Hidden -PassThru '
              '| Select-Object -ExpandProperty Id')
    launched = subprocess.run(['powershell.exe', '-NoProfile', '-Command', script],
                             capture_output=True, text=True, timeout=60)
    if launched.returncode or not launched.stdout.strip():
        raise RuntimeError('Administrator launch was not confirmed: ' + launched.stderr.strip())
    with (root / 'launch.json').open('x', encoding='utf-8') as stream:
        json.dump({'administrator_pid': launched.stdout.strip(), 'mode': mode}, stream)
    print('Administrator session started; waiting for result and restoration.', flush=True)
    deadline = time.monotonic() + 680
    result_path = root / 'admin_result.json'
    while not result_path.exists():
        if time.monotonic() > deadline:
            raise RuntimeError('Session result not confirmed; inspect logs and restoration watchdog')
        time.sleep(.5)
    # The controller creates the result only after restoration and owned-client cleanup.
    # Wait for its completed JSON write rather than treating file existence as completion.
    for attempt in range(20):
        try:
            admin = json.loads(result_path.read_text(encoding='utf-8'))
            break
        except json.JSONDecodeError:
            if attempt == 19:
                raise
            time.sleep(.1)
    restored = (not admin['failure'] and admin['root_held'] and admin['root_done']
                and admin['root_resumed_confirmed'] and admin['ubuntu_restore_returncode'] == 0)
    ubuntu_path = root / 'ubuntu_result.json'
    ubuntu = json.loads(ubuntu_path.read_text(encoding='utf-8')) if ubuntu_path.exists() else {}
    quality = ubuntu.get('summary') or {}
    report = {'restoration_confirmed': restored,
              'clock_quality_passed': (not ubuntu.get('error') and quality.get('quality_passed') is True) if quality else None,
              'clock_quality_evaluated': bool(quality), 'environment_error': ubuntu.get('error'),
              'output_dir': str(root), 'mode': mode, 'task_success': None}
    if mode == 'pilot':
        task_path = root / 'pilot/condition_00/evaluator/summary.json'
        task = json.loads(task_path.read_text(encoding='utf-8')) if task_path.exists() else {}
        report.update(task_category=task.get('category'), task_reason=task.get('reason'),
                      task_success=(task.get('category') == 'success') if task else None,
                      task_evaluated=bool(task))
    print(json.dumps(report, ensure_ascii=False), flush=True)
    if not restored or not report['clock_quality_passed']:
        return 2
    return 3 if mode == 'pilot' and not report['task_success'] else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True, help='A new Windows directory')
    parser.add_argument('--repo-root', type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument('--data-root', type=Path, default=Path('D:/AeroWAM-data'))
    parser.add_argument('--mode', choices=('probe', 'pilot'), default='probe')
    parser.add_argument('--method', choices=tuple(PILOT_CHECKPOINTS), default='joint_bc')
    parser.add_argument('--policy-seed', type=int, default=0)
    parser.add_argument('--execute', action='store_true', help='Request UAC, execute once, await restoration')
    args = parser.parse_args()
    root = prepare(args.output_dir, args.repo_root, args.data_root, args.mode, args.method, args.policy_seed)
    print('Prepared reviewable scripts: ' + str(root), flush=True)
    if args.execute:
        raise SystemExit(execute(root, args.mode))


if __name__ == '__main__':
    main()
