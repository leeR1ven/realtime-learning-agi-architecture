"""Summarize recorded weight experiments without changing any brain or report."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path


def summarize(path):
    doc=json.loads(path.read_text(encoding='utf8'))
    last=doc['checks'][-1]
    groups=defaultdict(list)
    commands=defaultdict(list)
    for cue in last['cues']:
        first=cue['trace'][0]
        groups[tuple(first['thought_cells'])].append(cue['frequency'])
        commands[tuple(first['muscles'])].append(cue['frequency'])
    return dict(report=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        status=doc['status'],steps=doc['current_steps'],parameters=doc['parameters'],
        movement=doc.get('movement','Older report: see independent physical audit'),
        old_content_retention=last['old_saved_content_retention'],
        known_cues=last['known_cue_count'],initial_cue_content=last['known_cue_initial_correct'],
        final_same_event_content_diagnostic=last['known_cue_first_final_correct'],
        first_frame_distinct_thought_patterns=len(groups),
        identical_first_frame_thought_groups=list(groups.values()),
        first_frame_distinct_motor_commands=len(commands),
        excitation_peak=last['pfc_excitatory_current_max'],inhibitory_strength=last['pfc_inhibitory_strength'],
        weight_statistics=last.get('pfc_weight_statistics'),
        physical_control_p95_ms=doc['physical_control_p95_ms'],
        inference_limit='Distinct neural responses and same-event content are diagnostics, not learned navigation or AGI scores.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reports',type=Path,nargs='+')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    rows=[summarize(path) for path in args.reports]
    args.output.write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps([{k:v for k,v in row.items() if k not in
        ('weight_statistics','identical_first_frame_thought_groups','inference_limit')} for row in rows],ensure_ascii=True))
