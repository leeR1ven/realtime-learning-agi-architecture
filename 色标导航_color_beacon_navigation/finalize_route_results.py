"""Tighten the predeclared first-passage timing check from saved trajectories.

This does not rerun, train, select, replace or discard any trial. The original
output and executing source hashes remain available for independent inspection.
"""
from pathlib import Path
import argparse
import hashlib
import json

from route_switch_experiment import write_report

HERE = Path(__file__).resolve().parent


def strict_score(row, cue_end):
    first = next((p for p in row['passages'] if p['direction'] == 'left_to_right'), None)
    last = (row['arrival_route_score'] or {}).get('passage')
    expected = row['expected_route']
    first_after = bool(first and first['time'] >= cue_end-1e-8)
    last_after = bool(last and last['time'] >= cue_end-1e-8)
    score = bool(row['correct_arrival'] and first_after and last_after
                 and first['route'] == expected and row['goal_preserved'])
    return first, first_after, score


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report', nargs='?', type=Path,
                        default=HERE/'results'/'route_switch_evaluation.json')
    args = parser.parse_args()
    raw = args.report.read_bytes()
    report = json.loads(raw)
    assert report['status'] == 'complete', 'Wait for the fixed experiment to finish.'
    if 'scoring_correction' in report:
        print('Already finalized; original evidence retained.')
        return
    original = args.report.with_name(args.report.stem+'.original.json')
    if original.exists():
        assert original.read_bytes() == raw, 'Refuse to replace distinct original evidence.'
    else:
        original.write_bytes(raw)
    changes = []
    for pair in report['pairs']:
        for branch, row in pair['branches'].items():
            first, first_after, score = strict_score(row, pair['cue_end_time'])
            if score != row['strict_success']:
                changes.append({'seed': pair['seed'], 'initial_route': pair['initial_route'],
                                'branch': branch, 'before': row['strict_success'], 'after': score})
            row['first_forward_passage'] = first
            row['first_passage_after_cue'] = first_after
            row['strict_success'] = score
        pair['pair_success'] = bool(pair['valid_prefix'] and all(
            pair['branches'][b]['strict_success'] for b in ('stay', 'switch')))
    report['scoring_correction'] = {
        'reason': 'Predeclared FIRST passage must also be after cue ends, not only last passage before arrival.',
        'raw_output': str(original.resolve()), 'raw_output_sha256': hashlib.sha256(raw).hexdigest(),
        'changed_branches': changes, 'all_trials_retained': True, 'new_rollouts': 0,
        'source_sha256': {name: hashlib.sha256((HERE/name).read_bytes()).hexdigest()
                          for name in ('finalize_route_results.py', 'route_state_digest.py')}}
    write_report(args.report, report)
    print(json.dumps({'summary': report['summary'], 'changed_branches': changes}, ensure_ascii=False))


if __name__ == '__main__':
    main()
