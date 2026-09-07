"""Parameter-only audit of the original motor encoder/reverse Hebbian table."""
from pathlib import Path
import json
import time
import numpy as np

from sensory_adapter import SensoryAdapter
from vocal_body import VocalBody


def run():
    started = time.perf_counter()
    rng = np.random.default_rng(719023)
    commands = rng.uniform([.6, .05, .05, .05], [.95, .95, .95, .95], (32, 4))
    # The calibration motor labels are actual executed body controls. This is
    # an isolated peripheral parameter audit, not extra memories in a life.
    body = VocalBody()
    for command in commands:
        assert np.isfinite(body.render_frame(command)).all()
    rows = []
    for levels in (40, 64):
        for threshold in (.35, .42, .48, .52, .58):
            for window in (0, 4, 8, 12, 20, 40):
                adapter = SensoryAdapter(motor_levels=levels, motor_threshold=threshold,
                                         motor_reverse_window=window)
                codes, initial_errors = [], []
                for i, command in enumerate(commands):
                    code = adapter.learn_executed(command)
                    codes.append(code)
                    if i == 15:
                        initial_errors = np.abs(np.stack([adapter.decode_motor(c) for c in codes]) - commands[:16])
                decoded = np.stack([adapter.decode_motor(c) for c in codes])
                errors = np.abs(decoded - commands)
                rows.append({'motor_levels': levels, 'motor_threshold': threshold,
                             'motor_reverse_window': window,
                             'mean_absolute_motor_error': float(errors.mean()),
                             'mean_tension_error_hz': float(errors[:, 1].mean() * 260.),
                             'p95_absolute_motor_error': float(np.percentile(errors, 95)),
                             'old_mean_absolute_error_before_new': float(initial_errors.mean()),
                             'old_mean_absolute_error_after_new': float(errors[:16].mean()),
                             'distinct_motor_codes': len({tuple(np.flatnonzero(c)) for c in codes}),
                             'reverse_connections': adapter.motor_reverse.条数()})
    rows.sort(key=lambda row: (row['mean_absolute_motor_error'], row['p95_absolute_motor_error']))
    baseline = next(row for row in rows if row['motor_levels'] == 40 and row['motor_threshold'] == .52 and row['motor_reverse_window'] == 20)
    report = {'scope': 'Original motor parameters only; no architecture or default life changes.',
              'physical_actions': len(commands), 'candidates': len(rows), 'seed': 719023,
              'baseline': baseline, 'best_by_mean_motor_error': rows[0], 'rows': rows,
              'wall_seconds': time.perf_counter()-started}
    path = Path(__file__).resolve().parent / 'results' / 'vocal_motor_resolution_audit.json'
    path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({k: report[k] for k in ('baseline', 'best_by_mean_motor_error', 'wall_seconds')}, indent=2))


if __name__ == '__main__':
    run()
