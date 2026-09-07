"""Recount stored-content diagnostics from raw original tables and saved cells.

No brain execution or learning. A different recalled event is not classified as
incorrect navigation: goal/route correctness requires a separate behavioral test.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from checkpoint import load_tree
from audit_continuous_learning import _sha, corrected_scores


def audit(report_path):
    report_path=Path(report_path)
    report=json.loads(report_path.read_text(encoding='utf-8'))
    assert report['status']=='complete'
    previous_path=Path(report['previous_report'])
    previous=json.loads(previous_path.read_text(encoding='utf-8'))
    assert previous['status']=='complete'
    checkpoint=Path(report['complete_life_checkpoint'])
    paths=(report_path,previous_path,checkpoint,Path(report['base_life']))
    hashes={str(path):_sha(path) for path in paths}
    assert hashes[str(checkpoint)]==report['complete_life_sha256']
    assert hashes[str(paths[-1])]==report['base_sha256']
    if previous.get('complete_life_sha256'):
        assert previous['complete_life_sha256']==report['base_sha256']
    state=load_tree(checkpoint)
    threshold=state['brain']['config']['memory_threshold']
    references=previous['catalog']
    retention={}
    mismatches=[]
    for name in ('visual','audio','motor'):
        table=state['brain']['runtime']['memories'][name]['time_to_feature']
        by_time={int(t):i for i,t in enumerate(table['sources'])}
        row_result=dict(correct=0,total=0,empty_references=0,empty_preserved=0,
                        minimum_expected_weight=float('inf'))
        for stamp, reference in references.items():
            expected=set(reference[name])
            assert report['catalog'][stamp][name]==reference[name]
            row=by_time.get(int(stamp))
            if row is None:
                weights={}
            else:
                a,b=int(table['offsets'][row]),int(table['offsets'][row+1])
                weights={int(cell):float(weight) for cell,weight in
                         zip(table['targets'][a:b],table['weights'][a:b])}
            actual={cell for cell,weight in weights.items() if weight>=threshold}
            if expected:
                row_result['total']+=1;row_result['correct']+=int(actual==expected)
                row_result['minimum_expected_weight']=min(row_result['minimum_expected_weight'],
                    min(weights.get(cell,0.) for cell in expected))
            else:
                row_result['empty_references']+=1;row_result['empty_preserved']+=int(not actual)
            if actual!=expected:
                mismatches.append(dict(modality=name,time=int(stamp),missing=sorted(expected-actual),extra=sorted(actual-expected)))
        retention[name]=row_result
        assert {key:row_result[key] for key in ('correct','total')}==report['checks'][-1]['old_saved_content_retention'][name]
    scores=corrected_scores(report,report['catalog'])
    assert all(not check['changed_scores'] for check in scores)
    index=state['brain']['runtime']['index']
    assert np.all(np.diff(index['times'])>0), 'Historical prefix vote audit requires this unwrapped chronological run'
    index_rows=[(int(stamp),set(map(int,index['features'][int(index['offsets'][i]):int(index['offsets'][i+1])])))
                for i,stamp in enumerate(index['times'])]
    diagnostics=[]
    global_thought_patterns=set()
    for check in report['checks']:
        cues=check['cues']
        thoughts={tuple(sorted(q['trace'][0]['thought_cells'])) for q in cues}
        all_thoughts={tuple(sorted(t['thought_cells'])) for q in cues for t in q['trace']}
        global_thought_patterns.update(all_thoughts)
        final_visual={tuple(sorted(q['trace'][0]['score']['final']['cells'])) for q in cues}
        actions={tuple(q['trace'][0]['muscles']) for q in cues}
        all_actions={tuple(t['muscles']) for q in cues for t in q['trace']}
        votes=[]
        for pattern in thoughts:
            cue_set=set(pattern)
            winner,best_votes,best_key=None,0,(-1,-1)
            for stamp,stored in index_rows:
                if stamp>=check['clock']:
                    continue
                count=len(stored&cue_set)
                if not count:
                    continue
                key=(count,-abs(len(stored)-len(cue_set)))
                if key>best_key:
                    winner,best_votes,best_key=stamp,count,key
            ratio=best_votes/len(cue_set) if cue_set else 0.
            accepted=winner if ratio>=report['brain_config']['index_threshold'] else None
            associated=[q for q in cues if tuple(sorted(q['trace'][0]['thought_cells']))==pattern]
            assert all(q['trace'][0]['score']['final']['time']==accepted for q in associated)
            votes.append(dict(frequencies=[q['frequency'] for q in associated],thought_cell_count=len(pattern),
                thought_cells_sha256=hashlib.sha256(np.asarray(pattern,dtype=np.int64).tobytes()).hexdigest(),
                best_available_time=winner,best_votes=best_votes,relative_vote=ratio,
                relative_threshold=report['brain_config']['index_threshold'],accepted_time=accepted))
        diagnostics.append(dict(stage=check['stage'],cue_count=len(cues),
            distinct_first_thought_sets=len(thoughts),distinct_first_final_visual_sets=len(final_visual),
            distinct_first_muscle_vectors=len(actions),
            first_thought_cell_counts=sorted({len(pattern) for pattern in thoughts}),
            distinct_all_probe_frame_thought_sets=len(all_thoughts),
            distinct_all_probe_frame_muscle_vectors=len(all_actions),
            unique_first_muscle_vectors=[list(action) for action in sorted(actions)],
            all_probe_final_indices_none=all(t['score']['final']['time'] is None for q in cues for t in q['trace']),
            all_probe_final_visual_cells_empty=all(not t['score']['final']['cells'] for q in cues for t in q['trace']),
            independent_original_index_vote=votes,
            first_final_content_nonmatches=[dict(frequency=q['frequency'],expected_color=q['color'],
                time=q['trace'][0]['score']['final']['time'],recalled_colors=q['trace'][0]['score']['final']['colors'])
                for q in cues if not q['trace'][0]['score']['final']['correct']]))
    assert hashes=={path:_sha(path) for path in hashes}
    result=dict(status='complete',source_report=str(report_path),unchanged_input_sha256=hashes,
        baseline_reference_count=len(references),reference_origin='Previous complete report catalog; raw earlier entries retained without relabelling.',
        old_content_retention_from_actual_CSR=retention,old_content_mismatches=mismatches,
        same_event_content_recount=scores,cue_state_distinctness=diagnostics,
        distinct_thought_patterns_across_all_saved_probe_frames=len(global_thought_patterns),
        index_vote_audit_method='Original set-intersection votes/gap/insertion-order rule; each historical check uses its append-only index prefix before the recorded clock, with no ring wrap.',
        no_model_execution_or_learning=True,
        limit='Neuron-set/content identity and action diversity only; none establishes correct goal-directed navigation or a failure of learned shortcuts.')
    output=report_path.with_name(report_path.stem+'_content_audit.json')
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(output=str(output),retention=retention,last=diagnostics[-1],sha256=_sha(output)),ensure_ascii=False))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('report',type=Path)
    audit(parser.parse_args().report)
