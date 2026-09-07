"""Verify the production reflex equals the independently tested candidate."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from audit_reflex_contact import directional_turning_reflex, generic_direction_checks
from audit_continuous_learning import _load_runner, _sha, tree_digest


def same_action(first, second):
    if first is None or second is None:
        return first is None and second is None
    return np.array_equal(first, second)


def audit():
    runner = _load_runner()
    from contact_reflex import contact_reflex
    here = Path(__file__).parent
    sources = {name: _sha(here/name) for name in ('brain.py','contact_reflex.py','audit_reflex_contact.py')}
    template = dict(observation=dict(touch=np.zeros(4), pain=0., ray_distances=np.full(31,2.),
        ray_colors=np.zeros((31,3)),velocity=np.zeros(2),angular_velocity=0.),waveform=np.zeros(800))
    cases=[]
    for side in range(4):
        for openness in ('left','right','equal'):
            packet=copy.deepcopy(template)
            packet['observation']['touch'][side]=1.
            if openness=='left': packet['observation']['ray_distances'][19:]=4.
            if openness=='right': packet['observation']['ray_distances'][:12]=4.
            cases.append((f"touch_{side}_open_{openness}",packet))
    for value in (0.,.019999,.02,.020001,1.):
        packet=copy.deepcopy(template);packet['observation']['pain']=value
        cases.append((f"pain_only_{value}",packet))
    for value in (.219999,.22,.220001):
        packet=copy.deepcopy(template);packet['observation']['ray_distances'][15]=value
        cases.append((f"front_depth_{value}",packet))
    for value in (.019999,.02,.020001):
        packet=copy.deepcopy(template);packet['observation']['touch'][:]=value
        cases.append((f"equal_touch_boundary_{value}",packet))
    rng=np.random.default_rng(570011)
    for index in range(256):
        packet=copy.deepcopy(template)
        packet['observation'].update(touch=rng.uniform(0,1,4),pain=float(rng.uniform(0,1)),
            ray_distances=rng.uniform(0,12.81,31),ray_colors=rng.uniform(0,1,(31,3)),
            velocity=rng.uniform(-2,2,2),angular_velocity=float(rng.uniform(-3,3)))
        packet['waveform']=rng.uniform(-.8,.8,800)
        cases.append((f"ordinary_sensor_mixture_{index}",packet))
    named=[]
    enabled=SimpleNamespace(flags={'reflex':True})
    disabled=SimpleNamespace(flags={'reflex':False})
    for name, packet in cases:
        before=tree_digest(packet)
        expected=directional_turning_reflex(packet)
        actual=contact_reflex(packet)
        assert same_action(expected,actual),name
        assert same_action(actual,runner.OriginalBrain._reflex(enabled,packet)),name
        assert runner.OriginalBrain._reflex(disabled,packet) is None,name
        assert tree_digest(packet)==before,name
        if not name.startswith('ordinary_sensor_mixture_'):
            named.append(dict(case=name,muscles=None if actual is None else actual.tolist()))
    assert sources=={name:_sha(here/name) for name in sources}
    report=dict(status='passed',source_sha256=sources,cases=len(cases),
        production_equals_frozen_candidate=True,brain_delegates_same_output=True,
        disabled_reflex_returns_none_all_cases=True,inputs_unchanged=True,
        named_cases=named,generic_force_and_turn_signs=generic_direction_checks(),
        new_brain_state_or_timer=False,no_training_or_checkpoint_write=True,
        limit='Integration equivalence of an innate contact controller; no learned navigation conclusion.')
    output=here/'results'/'contact_reflex_integration_audit.json'
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(output=str(output),cases=len(cases),status='passed',sha256=_sha(output)),ensure_ascii=False))


if __name__=='__main__':
    audit()
