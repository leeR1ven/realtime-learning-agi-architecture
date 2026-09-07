"""行为验收：留出、去教学、断线消融、融合、恢复和实时预算。"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import numpy as np
from brain import Brain, tone, DT, FREQUENCIES, SAMPLE_RATE
from 本能区_instinct import 本能区
from world import World
from checkpoint import merge_checkpoints, load_checkpoint

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / 'results'
DEFAULT_MAPPING = {0: 1, 1: 3, 2: 4, 3: 0}


def teach(brain, mapping, repetitions=20):
    for sound, action in mapping.items():
        brain.reset_activity()
        world = World()
        world.reset(position=(4, 3))
        for _ in range(repetitions):
            muscles, emitted, info = brain.step(world.observe(), tone(sound),
                stimulation=np.eye(5)[action], pleasant=1., mouth_stimulation=np.eye(4)[sound])
            world.step(muscles, DT)
    brain.reset_activity()


def query(brain, mapping, *, pfc=True, memory=True, noise=0., amplitude=.7, phase=0.):
    rng = np.random.default_rng(741)
    rows = []
    world = World()
    for sound, expected in mapping.items():
        brain.reset_activity()
        wave = amplitude * np.sin(2*np.pi*FREQUENCIES[sound]*np.arange(800)/SAMPLE_RATE + phase)
        wave += rng.normal(0, noise, 800)
        # 没有运动刺激、没有奖励，所有学习开关仍开。
        muscles, emitted, info = brain.step(world.observe(), wave,
            pfc_enabled=pfc, memory_enabled=memory)
        rows.append({'tone': sound, 'expected_action': expected, 'actual_action': info['action'],
                     'correct': info['action'] == expected, 'emitted': info['emitted'],
                     'muscles': muscles.tolist(), 'episodic': info['episodic_motor'],
                     'temporal': info['temporal_motor']})
    return rows


def encounter(brain, wall=(4., 1.4, 4.2, 4.6), *, frames=80, learning=True, color=(.35,.35,.35)):
    brain.reset_activity()
    world = World()
    world.reset(position=(1.2, 3), walls=[{'bounds': wall, 'color': color}])
    contacts = 0
    first_contact = None
    path = []
    total_impulse = 0.
    latencies = []
    for i in range(frames):
        start = time.perf_counter()
        muscles, _, info = brain.step(world.observe(), plasticity=learning)
        obs = world.step(muscles, DT)
        latencies.append((time.perf_counter() - start) * 1000)
        if obs['contact']:
            contacts += 1
            first_contact = i if first_contact is None else first_contact
        total_impulse += obs['contact_impulse']
        path.append(world.position.tolist())
    return {'contact_frames': contacts, 'first_contact_frame': first_contact,
            'impulse': total_impulse, 'path': path, 'final_position': world.position.tolist(),
            'p95_ms': float(np.percentile(latencies, 95)),
            'danger_weights': brain.danger_weights.tolist()}


def probe_cry_instinct():
    """全新脑(零可塑连接)撞墙即本能哭喊：哭不是学来的。"""
    brain = Brain()
    world = World()
    world.reset(position=(1.2, 3), walls=[{'bounds': (4., 1.4, 4.2, 4.6)}])
    cries = []
    for i in range(150):
        muscles, emitted, info = brain.step(world.observe(), stimulation=np.eye(5)[1], plasticity=False)
        world.step(muscles, DT)
        if info['cried']:
            cries.append(i)
    return {'cry_frames': cries, 'echo_zero': bool(not brain.echo_weights.any()),
            'audio_zero': bool(not brain.audio_weights.any()),
            'pfc_edges_zero': brain.pfc.连接数() == 0,
            'danger_zero': bool(not brain.danger_weights.any())}


def probe_pfc_takeover():
    """未学时本能前进占优；教学音调到静息后，本能衰减、动作由前额叶学习接管。"""
    fresh = Brain()
    world = World()
    muscles, _, info = fresh.step(world.observe(), tone(0))
    naive_action, naive_gain = info['action'], info['instinct_gain']
    teach(fresh, {0: 0})
    fresh.reset_activity()
    muscles, _, info = fresh.step(World().observe(), tone(0))
    return {'naive_action': naive_action, 'naive_gain': naive_gain,
            'taught_action': info['action'], 'taught_gain': info['instinct_gain'],
            'audio_weight': float(fresh.audio_weights[0, 0])}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--soak-frames', type=int, default=3000)
    args = parser.parse_args()
    RESULTS.mkdir(exist_ok=True)
    report = {'scope': '二维推进身体、五束RGB/深度、四音调关联；不证明双足/自然语言/AGI', 'checks': {}}
    started = time.perf_counter()

    novice = Brain()
    report['collision_learning_experience'] = encounter(novice)
    novice.save(RESULTS / 'after_collision.npz')
    held_out = []
    for shift, height, color in [(-.7,.6,(.6,.2,.2)), (0.,1.,(.2,.6,.2)),
            (.6,1.8,(.2,.2,.6)), (-.4,2.3,(.6,.6,.2)), (.4,1.4,(.15,.15,.15)),
            (.9,.85,(.7,.7,.7))]:
        wall = (4 + shift, 3 - height, 4.18 + shift, 3 + height)
        before = encounter(Brain(), wall, learning=False, color=color)
        after = encounter(Brain.load(RESULTS / 'after_collision.npz'), wall, color=color)
        transferred = encounter(Brain.load(RESULTS / 'after_collision.npz'), wall, learning=False, color=color)
        novice_online = encounter(Brain(), wall, color=color)
        no_synapses = Brain.load(RESULTS / 'after_collision.npz')
        no_synapses.danger_weights[:] = 0
        ablated = encounter(no_synapses, wall, learning=False, color=color)
        held_out.append({'wall': list(wall), 'color': color, 'before': before, 'after': after,
                         'learned_frozen': transferred, 'novice_online': novice_online, 'ablated': ablated})
    report['collision_holdout'] = held_out
    before_hits = sum(x['before']['contact_frames'] for x in held_out)
    after_hits = sum(x['after']['contact_frames'] for x in held_out)
    report['checks']['collision_learning_changes_behavior'] = bool(after_hits < before_hits)
    report['checks']['collision_ablation_restores_hits'] = bool(sum(x['ablated']['contact_frames'] for x in held_out) > after_hits)
    report['checks']['collision_experience_transfers_without_new_learning'] = bool(sum(x['learned_frozen']['contact_frames'] for x in held_out) < before_hits)
    print('collision:', before_hits, '->', after_hits, flush=True)

    # 随机动作语义分配，防止把硬编码tone->action误当学习。
    mapping = {0: 4, 1: 0, 2: 1, 3: 3}
    language = Brain()
    report['audio_before'] = query(language, mapping)
    teach(language, mapping)
    language.save(RESULTS / 'audio_trained.npz')
    report['audio_after'] = query(Brain.load(RESULTS / 'audio_trained.npz'), mapping,
                                noise=.015, amplitude=.32, phase=.83)
    report['audio_no_pfc_no_memory'] = query(Brain.load(RESULTS / 'audio_trained.npz'), mapping,
                                           pfc=False, memory=False)
    pfc_only = Brain.load(RESULTS / 'audio_trained.npz')
    pfc_only.audio_weights[:] = 0
    report['audio_temporal_only'] = query(pfc_only, mapping, memory=False)
    report['audio_memory_only'] = query(Brain.load(RESULTS / 'audio_trained.npz'), mapping, pfc=False)
    report['checks']['audio_learns_arbitrary_mapping'] = all(x['correct'] for x in report['audio_after'])
    report['checks']['audio_ablation_loses_mapping'] = not all(x['correct'] for x in report['audio_no_pfc_no_memory'])
    report['checks']['original_pfc_has_causal_action_path'] = all(x['correct'] for x in report['audio_temporal_only'])
    report['checks']['original_time_memory_has_causal_action_path'] = all(x['correct'] for x in report['audio_memory_only'])
    report['checks']['autonomous_sound_response'] = all(x['tone'] in x['emitted'] for x in report['audio_after'])
    print('audio:', [(x['actual_action'],x['correct']) for x in report['audio_after']], flush=True)

    # ---- 出生本能区：固定直接投射(不经过前额叶) + 学习后的本能衰减 ----
    report['instinct_cry'] = probe_cry_instinct()
    cry_report = report['instinct_cry']
    report['checks']['instinct_pain_cry_without_learning'] = bool(
        cry_report['cry_frames'] and cry_report['echo_zero'] and cry_report['audio_zero']
        and cry_report['pfc_edges_zero'] and cry_report['danger_zero'])
    report['instinct_pfc_takeover'] = probe_pfc_takeover()
    takeover = report['instinct_pfc_takeover']
    report['checks']['learned_pfc_suppresses_instinct'] = bool(
        takeover['naive_gain'] > 0.9 and takeover['naive_action'] != 0
        and takeover['taught_action'] == 0 and takeover['taught_gain'] < 0.5)
    s = 本能区()
    mid = np.full(5, 3.0)
    out, _ = s.驱动(np.array([1., 0, 0, 0]), mid, 0., 0); ok_front = int(np.argmax(out)) == 2
    out, _ = s.驱动(np.array([0., 0, 1, 0]), mid, 0., 0); ok_left = int(np.argmax(out)) == 4
    out, _ = s.驱动(np.array([0., 0, 0, 1]), mid, 0., 0); ok_right = int(np.argmax(out)) == 3
    out, _ = s.驱动(np.array([1., 0, 0, 0]), np.array([4.9, 4.9, 2., 2., 2.]), 0., 0)
    ok_select = bool(out[4] > out[3])
    _, cry_on = s.驱动(np.zeros(4), mid, .8, 0)
    _, cry_off = s.驱动(np.zeros(4), mid, .1, 0)
    report['checks']['instinct_contact_escape_directions'] = bool(
        ok_front and ok_left and ok_right and ok_select and cry_on and not cry_off)
    print('instinct cry frames:', cry_report['cry_frames'][:3],
          '| takeover:', {k: v for k, v in takeover.items()}, flush=True)


    a, b = Brain(), Brain()
    teach(a, {0: 1, 1: 3}); teach(b, {2: 4, 3: 0})
    a.save(RESULTS / 'branch_a.npz'); b.save(RESULTS / 'branch_b.npz')
    merge_checkpoints([RESULTS / 'branch_a.npz', RESULTS / 'branch_b.npz'], RESULTS / 'fused.npz')
    fused = Brain.load(RESULTS / 'fused.npz')
    report['fusion_responses'] = query(fused, DEFAULT_MAPPING)
    aa, _ = load_checkpoint(RESULTS / 'branch_a.npz'); bb, _ = load_checkpoint(RESULTS / 'branch_b.npz')
    ff, _ = load_checkpoint(RESULTS / 'fused.npz')
    report['fusion_counts'] = {'a_events': len(aa['mem_event_tick']), 'b_events':len(bb['mem_event_tick']),
                               'fused_events':len(ff['mem_event_tick']), 'fused_edges':len(ff['mem_edge_time'])}
    report['checks']['fusion_preserves_both_skills'] = all(x['correct'] for x in report['fusion_responses'])
    report['checks']['fusion_preserves_all_memories'] = len(ff['mem_event_tick']) == len(aa['mem_event_tick']) + len(bb['mem_event_tick'])
    # 融合后继续经历真实碰撞，检验可塑性未被冻结。
    report['fusion_continued_learning'] = encounter(fused)
    report['checks']['fused_brain_still_learns'] = bool(np.any(fused.danger_weights > 0))
    fused.reset_activity(); fused.save(RESULTS / 'demo_brain.npz')

    # 读回后相同观察序列产生完全一致的动作与新权重。
    restored = Brain.load(RESULTS / 'demo_brain.npz')
    equal = True
    for i in range(20):
        obs = World().observe()
        x = tone(i % 4) if i % 3 else np.zeros(800)
        r1 = fused.step(obs, x); r2 = restored.step(obs, x)
        equal &= np.array_equal(r1[0], r2[0]) and np.array_equal(r1[1], r2[1])
    equal &= np.array_equal(fused.audio_weights, restored.audio_weights)
    equal &= np.array_equal(fused.danger_weights, restored.danger_weights)
    report['checks']['checkpoint_continuation_matches'] = bool(equal)
    motor_lesion = Brain.load(RESULTS / 'demo_brain.npz')
    motor_lesion.motor.reverse.矩阵[:] = False
    report['checks']['motor_connections_cause_motion'] = all(np.all(motor_lesion.motor.decode(i) == 0) for i in range(5))

    # 连续完整闭环压测，学习/思考/物理一起计时，不把仿真时长当墙钟时长。
    live = Brain.load(RESULTS / 'demo_brain.npz')
    world = World(walls=[(3., 1., 3.2, 2.5), (5., 3.4, 5.2, 5.)],
                  objects=[{'position':(6.5,2.5),'color':(1,0,0),'radius':.45}])
    times, trace, positions = [], [], []
    echo = np.zeros(800)
    for i in range(args.soak_frames):
        t = time.perf_counter()
        muscles, echo, info = live.step(world.observe(), echo)
        obs = world.step(muscles, DT)
        positions.append(world.position.copy())
        times.append((time.perf_counter() - t) * 1000)
        if i % 10 == 0:
            trace.append({'frame':i, 'position':world.position.tolist(), **info})
        if i and i % 1000 == 0:
            print('soak', i, 'p95ms', round(float(np.percentile(times[-1000:],95)),2), flush=True)
    report['performance'] = {'frames':args.soak_frames, 'simulated_seconds':args.soak_frames*DT,
        'compute_seconds':sum(times)/1000, 'p50_ms':float(np.percentile(times,50)),
        'p95_ms':float(np.percentile(times,95)), 'p99_ms':float(np.percentile(times,99)),
        'max_ms':max(times), 'over_100ms_frames':sum(x>100 for x in times),
        'memory_edges':live.last_info['memory_edges'], 'pfc_edges':live.last_info['pfc_edges'],
        'peak_thought_active': max(x['thought_active'] for x in trace),
        'path_displacement':float(np.linalg.norm(world.position - np.array([4.,3.]))) }
    report['performance']['total_distance_m'] = float(np.linalg.norm(np.diff(positions,axis=0),axis=1).sum())
    report['performance']['last_1000_frames_distance_m'] = float(np.linalg.norm(np.diff(positions[-1000:],axis=0),axis=1).sum())
    report['checks']['real_time_p95_under_100ms'] = report['performance']['p95_ms'] < 100
    report['checks']['all_world_values_finite'] = bool(np.isfinite(world.position).all())
    report['checks']['continues_moving_without_turning_in_place_forever'] = report['performance']['last_1000_frames_distance_m'] > 2.
    report['elapsed_seconds'] = time.perf_counter() - started
    report['passed'] = all(report['checks'].values())
    (RESULTS/'evaluation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    (RESULTS/'trace.json').write_text(json.dumps(trace,ensure_ascii=False),encoding='utf-8')
    print(json.dumps({'passed':report['passed'],'checks':report['checks'],'performance':report['performance']},ensure_ascii=False,indent=2))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
