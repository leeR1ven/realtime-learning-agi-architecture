"""One mechanism-derived gain change; preserve every unsuccessful condition."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from brain import OriginalBrain
from experience_environment import ExperienceEnvironment
from dynamic_calibration import hashes
from run_closed_loop import COLORS, TONES, plain, train
from validate_closed_loop import probe, summarize

HERE = Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=HERE/'results'/'recall_gain_validation.json')
    parser.add_argument('--index-threshold',type=float)
    parser.add_argument('--fresh-seed',type=int,default=2029)
    args=parser.parse_args()
    config=json.loads((HERE/'results'/'dynamic_best_tested_config.json').read_text(encoding='utf8'))
    config['recall_gain']=6.
    if args.index_threshold is not None:
        config['index_threshold']=args.index_threshold
    config_path=(HERE/'results'/'continuity_gain6_config.json' if args.index_threshold is None
                 else args.output.with_name(args.output.stem+'_config.json'))
    config_path.write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf8')
    source=hashes()
    report=dict(status='running',config=config,source_sha256=source,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        rationale=('On identical failure state, green cell2892 has recall3.5+E.3271 below threshold5.33146; silence removes current3.5. One candidate recall gain6 tests this measured loss, without a new memory/gate/equation.'
            if args.index_threshold is None else
            'Long blue failure preserves all49 blue-specific cells but rejects correct blue candidates at224/262=.855 below original index threshold.9. Test one lower original parameter.8 while keeping recall gain6; no changed index equation.'),
        protocol=f'Re-train with shown parameters. Seed2027 reproduction, new birth{args.fresh_seed} normal+permuted, 2026 longer experience. No selection from new-birth results.',
        cases=[])
    for name,seed,permutation,repeats in (
        ('failure_reproduction',2027,(0,1,2),2),
        ('new_birth',args.fresh_seed,(0,1,2),2),
        ('new_birth_permuted',args.fresh_seed,(2,0,1),2),
        ('longer_experience',2026,(0,1,2),6)):
        brain=OriginalBrain(dict(config,seed=seed))
        env=ExperienceEnvironment(seed=20260909)
        training=train(brain,env,repeats=repeats,permutation=permutation,hold_frames=8)
        snapshot=brain.snapshot()
        path=args.output.with_name(args.output.stem+'_'+name+'.npz')
        if path.exists(): raise FileExistsError(path)
        brain.save(path)
        case=dict(name=name,seed=seed,permutation=permutation,training=training,
            checkpoint=str(path),probes=[],cue_switches=[],unknown_sound_probes=[],amplitude_probes=[])
        for condition in ('full','no_recalled_drive','no_pfc_recurrence','no_disinhibition','no_motor_recall'):
            for i,color in enumerate(COLORS):
                case['probes'].append(probe(snapshot,env,training['catalog'],
                    frequency=TONES[permutation[i]],expected=color,condition=condition))
        for a,b in ((0,1),(1,2),(2,0)):
            case['cue_switches'].append(probe(snapshot,env,training['catalog'],
                frequency=TONES[permutation[a]],expected=COLORS[a],
                switch_frequency=TONES[permutation[b]],switch_expected=COLORS[b]))
        if name=='new_birth':
            for frequency in (777.,2047.):
                case['unknown_sound_probes'].append(probe(snapshot,env,training['catalog'],
                    frequency=frequency,expected=None,steps=1))
            for amplitude in (.3,.6):
                for i,color in enumerate(COLORS):
                    case['amplitude_probes'].append(probe(snapshot,env,training['catalog'],
                        frequency=TONES[i],expected=color,amplitude=amplitude,steps=1))
        case['summary']={condition:summarize([r for r in case['probes'] if r['condition']==condition])
            for condition in dict.fromkeys(r['condition'] for r in case['probes'])}
        case['switch_summary']=summarize(case['cue_switches'])
        report['cases'].append(case)
        assert hashes()==source
        args.output.write_text(json.dumps(plain(report),ensure_ascii=False,indent=2),encoding='utf8')
        print('GAIN6 '+name+' '+json.dumps(case['summary'])+' SWITCH '+json.dumps(case['switch_summary']),flush=True)
    accepted=all(c['summary']['full']['initial_correct']==3 and
        c['summary']['full']['final_correct_frames']==c['summary']['full']['frames'] and
        c['switch_summary']['final_correct_frames']==c['switch_summary']['frames'] for c in report['cases'])
    report.update(status='complete',all_tested_retention_and_switch_frames_correct=accepted,
        boundary='Associative visual-content maintenance and cue switching; not routes or speech or AGI. Recurrence necessity is tested separately and must not be inferred from success.')
    args.output.write_text(json.dumps(plain(report),ensure_ascii=False,indent=2),encoding='utf8')


if __name__=='__main__':
    main()
