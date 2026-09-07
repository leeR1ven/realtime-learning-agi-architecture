"""Counterfactual contact-reflex audit on copies of an actual life checkpoint.

No production brain, environment, or learned checkpoint is edited. The candidate
reads only the ordinary sensor packet. Coordinates and collision normals below
are evaluator-only evidence, never arguments to the candidate reflex.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import inspect
import json
from pathlib import Path
import types

import numpy as np

from audit_continuous_learning import _load_runner, _sha, tree_digest


def legacy_contact_reflex(packet):
    """Frozen pre-fix baseline, independent of future production brain edits."""
    observation = packet['observation']
    touched = max(float(observation['pain']), float(np.max(observation['touch']))) > .02
    if touched or np.min(np.asarray(observation['ray_distances'])[12:19]) < .22:
        return np.array([0., .5, 0., .5])
    return None


def directional_contact_reflex(packet):
    """Stateless candidate: front/back withdrawal, sides turn and move away.

    Touch order is front, back, left, right; action order is LF, LB, RF, RB.
    Pain without a touch direction requests zero drive, not arbitrary reverse.
    Original close-front depth caution is retained. No target/RGB label is read.
    """
    observation = packet['observation']
    touch = np.asarray(observation['touch'], float)
    if np.max(touch) > .02:
        actions = np.array([
            [0., .5, 0., .5],  # front contact: backwards
            [.5, 0., .5, 0.],  # back contact: forwards
            [.5, 0., .1, 0.],  # left contact: forward and clockwise
            [.1, 0., .5, 0.],  # right contact: forward and counterclockwise
        ])
        return actions[int(np.argmax(touch))].copy()
    if float(observation['pain']) > .02:
        return np.zeros(4)
    if np.min(np.asarray(observation['ray_distances'])[12:19]) < .22:
        return np.array([0., .5, 0., .5])
    return None


def directional_turning_reflex(packet):
    """Second stateless candidate: add a turn to front/back withdrawal.

    The turn chooses the more open side from ordinary forward visual distances;
    ties use a fixed born chirality. This is a safety motion, not a route plan.
    """
    result = directional_contact_reflex(packet)
    observation = packet['observation']
    touch = np.asarray(observation['touch'], float)
    if np.max(touch) > .02 and int(np.argmax(touch)) in (0, 1):
        distances = np.asarray(observation['ray_distances'])
        left_open, right_open = np.mean(distances[19:]), np.mean(distances[:12])
        turn_left = left_open > right_open
        if int(np.argmax(touch)) == 1:
            return np.array([.1, 0., .5, 0.]) if turn_left else np.array([.5, 0., .1, 0.])
        # During reverse travel the yaw sign must be reversed for the actual
        # retreat trajectory to bend toward the visually more open side.
        return np.array([0., .1, 0., .5]) if turn_left else np.array([0., .5, 0., .1])
    return result


def generic_direction_checks():
    """Sensor-only unit fixtures for force and turning signs; no world reset."""
    packet = dict(observation=dict(touch=np.zeros(4), pain=0.,
        ray_distances=np.full(31,2.), ray_colors=np.zeros((31,3)),
        velocity=np.zeros(2), angular_velocity=0.),waveform=np.zeros(800))
    rows=[]
    for touched_side in (0,1):
        for open_side in ('left','right'):
            o=packet['observation']
            o['touch'][:]=0.;o['touch'][touched_side]=1.
            o['ray_distances'][:]=1.
            o['ray_distances'][19:] = 4. if open_side=='left' else 1.
            o['ray_distances'][:12] = 4. if open_side=='right' else 1.
            action=directional_turning_reflex(packet)
            left=action[0]-action[1];right=action[2]-action[3]
            longitudinal=left+right;yaw=right-left
            assert longitudinal < 0 if touched_side==0 else longitudinal > 0
            # The sign of forward force times yaw gives the initial lateral
            # bending direction; reverse motion flips the required yaw sign.
            assert longitudinal*yaw > 0 if open_side=='left' else longitudinal*yaw < 0
            rows.append(dict(touch=('front','back')[touched_side],open_side=open_side,
                             muscles=action.tolist(),longitudinal_force_sign=float(np.sign(longitudinal)),
                             retreat_bending_side='left' if longitudinal*yaw>0 else 'right'))
    packet['observation']['touch'][:]=0.
    packet['observation']['pain']=1.
    assert np.array_equal(directional_turning_reflex(packet),np.zeros(4))
    return dict(front_back_retreat_toward_open_side=rows,
                nondirectional_residual_pain_requests_zero_drive=True)


def _trace_row(env, muscles, source):
    packet = env.sensor_packet()
    return dict(time=float(env.world.time), position=env.world.position.tolist(),
        heading=float(env.world.heading), speed=float(np.linalg.norm(env.world.velocity)),
        angular_velocity=float(env.world.angular_velocity),
        touch=packet['observation']['touch'].tolist(), pain=float(packet['observation']['pain']),
        muscles=np.asarray(muscles).tolist(), source=source,
        contact_normals=[row['normal'] for row in env.world.contacts])


def _summarize(env, start_position, start_path, start_contact, rows):
    return dict(net_displacement=float(np.linalg.norm(env.world.position-start_position)),
        path_length=float(env._path_length-start_path),
        contact_steps=int(env._contact_steps-start_contact),
        movement_steps=sum(row['speed']>.01 for row in rows),
        final_position=env.world.position.tolist(), final_heading=float(env.world.heading),
        final_wall_clearance=float(env.world.config.width-env.world.config.radius-env.world.position[0]),
        source_counts={name:sum(row['source']==name for row in rows) for name in sorted({r['source'] for r in rows})},
        trace=rows)


def audit(checkpoint, *, steps=100):
    runner = _load_runner()
    base = runner.LifeSession.load(checkpoint)
    input_sha = _sha(checkpoint)
    original_state = tree_digest(base.snapshot())
    start = base.environment.world.position.copy()
    heading = base.environment.world.heading
    forward = np.array([np.cos(heading), np.sin(heading)])
    initial_packet = base.environment.sensor_packet()
    old_command = legacy_contact_reflex(initial_packet)
    corrected_command = directional_contact_reflex(initial_packet)
    normal = np.asarray(base.environment.world.contacts[-1]['normal'])
    command_force = lambda action: (action[0]-action[1]+action[2]-action[3])*base.environment.world.config.max_muscle_force*forward
    evidence = dict(initial=_trace_row(base.environment, base.executed, 'saved_actual'),
        original_reflex_command=old_command.tolist(), directional_command=corrected_command.tolist(),
        wall_normal=normal.tolist(), original_force_world=command_force(old_command).tolist(),
        directional_force_world=command_force(corrected_command).tolist(),
        original_force_dot_outward_normal=float(command_force(old_command)@normal),
        directional_force_dot_outward_normal=float(command_force(corrected_command)@normal))
    assert evidence['original_force_dot_outward_normal'] < 0
    assert evidence['directional_force_dot_outward_normal'] > 0
    physical = {}
    # Pure-body counterfactuals isolate escape from memory/policy changes.
    for name, action in (('continue_original_reverse',old_command), ('withdraw_forward',corrected_command),
                        ('clockwise_then_forward',np.array([.3,0.,0.,.3])),
                        ('counterclockwise_then_forward',np.array([0.,.3,.3,0.]))):
        env = copy.deepcopy(base.environment)
        initial_path, initial_contact = env._path_length, env._contact_steps
        rows = []
        for frame in range(12):
            applied = action if frame < 5 or 'then' not in name else corrected_command
            env.step(applied)
            rows.append(_trace_row(env,applied,name))
        physical[name] = _summarize(env,start,initial_path,initial_contact,rows)
    assert physical['continue_original_reverse']['path_length'] < 1e-8
    assert physical['withdraw_forward']['final_wall_clearance'] > .1
    assert physical['withdraw_forward']['contact_steps'] < 3

    # Full frozen-brain boundary: every action after the first is independently
    # chosen using updated actual sensors; recalled old muscles remain present.
    # Both arms discard the old prepared action, so the intervention is applied
    # to the same fresh observation. No learned weight or time is reset.
    full = {}
    for name, candidate in (('original_reflex',legacy_contact_reflex), ('directional',directional_contact_reflex),
                            ('directional_with_turn',directional_turning_reflex)):
        session = runner.LifeSession.load(checkpoint)
        session.brain.prepared = None
        session.brain.pending = None
        if candidate is not None:
            def patched(self, packet, method=candidate):
                return method(packet) if self.flags['reflex'] else None
            session.brain._reflex = types.MethodType(patched, session.brain)
        env, brain = session.environment, session.brain
        initial_path, initial_contact = env._path_length, env._contact_steps
        rows=[]
        packet=env.sensor_packet()
        before_weights=runner.frozen_digest(brain)
        for frame in range(steps):
            muscles,info=brain.observe_then_act(packet,learning=False)
            packet=env.step(muscles)
            brain.observe_outcome(packet,muscles,terminal=frame==steps-1)
            rows.append(_trace_row(env,muscles,info['source']))
        assert runner.frozen_digest(brain)==before_weights
        full[name]=_summarize(env,start,initial_path,initial_contact,rows)
        full[name]['weights_unchanged']=True
    assert tree_digest(base.snapshot())==original_state
    assert _sha(checkpoint)==input_sha
    result=dict(status='complete',checkpoint=str(checkpoint),checkpoint_sha256=input_sha,
        candidate_source_sha256=_sha(__file__),
        frozen_legacy_function_sha256=hashlib.sha256(inspect.getsource(legacy_contact_reflex).encode()).hexdigest(),
        generic_direction_checks=generic_direction_checks(),
        evidence=evidence,physical_counterfactuals=physical,
        full_brain_sensor_closed_loop=full,full_brain_steps_per_condition=steps,
        production_brain_or_environment_modified=False,
        no_pose_or_velocity_injection=True,no_teacher_or_target_route=True,
        limit='Local contact-escape causality only; 100 frozen-brain frames do not establish learned navigation or long-term stability.')
    output=Path(__file__).parent/'results'/'reflex_contact_audit.json'
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    return result,output


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('checkpoint',type=Path)
    parser.add_argument('--steps',type=int,default=100)
    args=parser.parse_args()
    result,output=audit(args.checkpoint,steps=args.steps)
    print(json.dumps({'output':str(output),
        'full_loop':{name:{k:row[k] for k in ('path_length','net_displacement','contact_steps','final_wall_clearance','source_counts')}
                     for name,row in result['full_brain_sensor_closed_loop'].items()}},ensure_ascii=False))
