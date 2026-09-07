"""Promote the audited descendant only if the user's original life is unchanged."""
import hashlib
import json
from pathlib import Path
import shutil
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path.insert(0,str(HERE))
from audit_continuous_learning import _load_runner,tree_digest


def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    runner=_load_runner()
    report_path=HERE/'results'/'active_body_additive_600.json'
    report=json.loads(report_path.read_text(encoding='utf8'))
    candidate=report_path.with_suffix('.npz')
    if report['status']!='complete' or digest(candidate)!=report['complete_life_sha256']:
        raise ValueError('Audited descendant checkpoint differs from complete report')
    lineage=[]
    cursor=report
    while True:
        base=Path(cursor['base_life'])
        if not base.is_absolute(): base=ROOT/base
        if digest(base)!=cursor['base_sha256']:
            raise ValueError('An ancestor life has changed; do not overwrite it')
        lineage.append(dict(base_life=str(base),base_sha256=cursor['base_sha256']))
        previous=cursor.get('previous_report')
        if not previous: break
        previous=Path(previous)
        if not previous.is_absolute(): previous=ROOT/previous
        older=json.loads(previous.read_text(encoding='utf8'))
        assert older['status']=='complete'
        assert older['checks'][-1]['clock']==cursor['starts']['clock']
        cursor=older
    target=HERE/'results'/'living_original.npz'
    assert base.resolve()==target.resolve()
    backup=HERE/'results'/'living_before_continuous_learning.npz'
    assert target.resolve().is_relative_to(ROOT.resolve())
    assert backup.resolve().is_relative_to(ROOT.resolve())
    before=digest(target)
    session=runner.LifeSession.load(candidate)
    assert session.brain.config.pfc_weight_ceiling is None
    assert all(session.brain.flags.values())
    assert session.brain.runtime.clock.时间激活.sum()==1
    if backup.exists():
        assert digest(backup)==before,'Existing rollback differs; preserve it'
    else:
        shutil.copy2(target,backup)
    expected=tree_digest(session.snapshot())
    assert digest(target)==before,'User life changed during validation'
    session.save(target)
    restored=runner.LifeSession.load(target)
    assert tree_digest(restored.snapshot())==expected
    record=dict(promoted=True,report=str(report_path),candidate=str(candidate),
        candidate_sha256=digest(candidate),target=str(target),target_sha256=digest(target),
        previous_sha256=before,rollback=str(backup),lineage=lineage,
        all_flags_enabled=True,single_time_neuron_active=True,
        actual_control_steps=restored.control_steps,
        current_time_cell=int(restored.brain.runtime.clock.当前时间),
        physical_time=float(restored.environment.world.time),
        full_selected_life_equal_after_save_load=True,
        selected_learning_rule='Original additive with audited decay parameters; nonlinear options remain disabled')
    (HERE/'results'/'continuous_promotion.json').write_text(
        json.dumps(record,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(record,ensure_ascii=True))


if __name__=='__main__': main()
