"""Explicit experimental PFC extension, NOT an AGI or unchanged-original claim.

OriginalRuntime remains the single chronological multimodal memory. New learned
sensory assemblies form by novelty/competition. Experienced source + actual
muscle + successor establish reciprocal relational connections, with references
to the original time containing that actual muscle. Goal excitation is recalled
through the ORIGINAL time index, then spread on these learned connections.

The discounted maximum propagation below is an ENGINEERED planning prior
(equivalent to deterministic value propagation on experienced transitions).
It is not evidence that the original spiking PFC spontaneously acquired a
planner. It permits testing missing relational capacity and body interfaces.
No coordinates, labels, graph from environment, teacher actions or route
answers are accepted. Complete traces expose the learned route relation used.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
import numpy as np

from sensory_adapter import SensoryAdapter
from original_runtime import OriginalRuntime
from brain import packet_digest


@dataclass
class DevelopmentalConfig:
    seed: int = 20260906
    capacity: int = 128
    assembly_size: int = 4
    visual_samples: int = 9
    visual_ray_count: int = 31
    match_threshold: float = .985
    discount: float = .94
    thought_steps: int = 96
    connection_increment: float = .25
    connection_ceiling: float = 1.


class DevelopmentalNavigator:
    def __init__(self, config=None):
        self.config = config if isinstance(config, DevelopmentalConfig) else DevelopmentalConfig(**(config or {}))
        c = self.config
        if (type(c.capacity) is not int or not 4 <= c.capacity <= 4096
                or type(c.assembly_size) is not int or not 1 <= c.assembly_size <= 32
                or type(c.visual_samples) is not int or not 1 <= c.visual_samples <= 512
                or type(c.visual_ray_count) is not int or not 8 <= c.visual_ray_count <= 512
                or type(c.thought_steps) is not int or not 1 <= c.thought_steps <= 8192):
            raise ValueError('Invalid capacity or computation budget')
        if not 0 < c.discount < 1 or not 0 < c.match_threshold <= 1:
            raise ValueError('Invalid matching/propagation parameter')
        if not 0 < c.connection_increment <= c.connection_ceiling:
            raise ValueError('Invalid saturating plasticity parameters')
        self.adapter = SensoryAdapter(seed=c.seed, visual_threshold=.58, audio_threshold=.52,
                                      motor_threshold=.52, motor_levels=40, contrast_to_silence=True,
                                      ray_count=c.visual_ray_count)
        self.visual_samples = c.visual_samples
        self.visual_width = self.visual_samples * self.adapter.levels * 3 * 2
        if self.visual_width <= 0:
            raise ValueError('visual sample width must be positive')
        self.audio_offset = c.capacity * c.assembly_size
        self.runtime = OriginalRuntime(self.adapter.widths['visual'], self.adapter.widths['audio'], 4,
            self.audio_offset + self.adapter.widths['audio'], time_neurons=32768,
            pfc_parameters={'门槛': 2., '目标上限': 350},
            pfc_connection_parameters={'强化量': .0001, '衰减率': .9995, '警戒线': 1,
                                       '衰减间隔': 25, '消失下限': .0000001},
            memory_connection_parameters={'衰减率': 1.})
        self.prototypes = np.empty((0, self.visual_width), bool)
        self.visits = []
        # Connection row: source assembly, actual motor channel, successor,
        # Hebbian strength, source time, successor time (actual motor reference).
        self.edges = {}
        self.attempts = np.zeros((c.capacity, 4), np.int64)
        self.inhibition = np.zeros((c.capacity, 4))
        self.previous_place = None
        self.previous_time = None
        self.last_action = np.zeros(4)
        self.goal = None
        self.goal_time = None
        self.goal_activity = np.zeros(c.capacity)
        self.rng = np.random.default_rng(c.seed)
        self.learning_steps = 0
        self.pending = None
        self.last_trace = {}

    def _view(self, encoded):
        # Fixed camera receptive field: the downward floor region. This is
        # an explicit perceptual prior, never an environment place identifier.
        rgb = encoded['input_arrays']['rgb'].astype(bool, copy=True)
        width = self.visual_width if len(self.prototypes) else self.visual_width
        if rgb.size < width:
            rgb = np.pad(rgb, (0, width - rgb.size))
        elif rgb.size > width:
            rgb = rgb[:width]
        return rgb.astype(bool)

    def _place(self, view, learning):
        scores = (np.mean(self.prototypes == view, axis=1) if len(self.prototypes) else np.empty(0))
        if len(scores) and scores.max() >= self.config.match_threshold:
            index = int(np.argmax(scores))
        elif learning and len(self.prototypes) < self.config.capacity:
            index = len(self.prototypes)
            self.prototypes = np.vstack([self.prototypes, view])
            self.visits.append(0)
        else:
            return None, float(scores.max()) if len(scores) else 0.
        if learning:
            self.visits[index] += 1
        return index, float(scores[index]) if index < len(scores) else 1.

    def _cue(self, packet, encoded):
        if np.sqrt(np.mean(np.asarray(packet['waveform'])**2)) < .02:
            return None
        query = np.zeros(self.runtime.widths['pfc'], bool)
        query[self.audio_offset:] = encoded['audio']
        recalled = self.runtime.recall_from_pfc(query, .8, .5)
        if recalled['time'] is None:
            self.goal = self.goal_time = None
            return dict(recognized=False, ratio=recalled['ratio'])
        # Recover the learnt assembly at the same shared time. No independent
        # sound->goal label table is maintained.
        stored = self.runtime.index.时间到输出.get(recalled['time'], set())
        groups = [i // self.config.assembly_size for i in stored if i < self.audio_offset]
        if not groups:
            self.goal = self.goal_time = None
            return dict(recognized=False, ratio=recalled['ratio'])
        self.goal = int(np.bincount(groups).argmax())
        self.goal_time = int(recalled['time'])
        return dict(recognized=True, time=self.goal_time, assembly=self.goal, ratio=recalled['ratio'])

    def _propagate(self, enabled=True):
        c = self.config
        values = np.zeros(c.capacity)
        if self.goal is None or not enabled:
            self.goal_activity = values.copy()
            return values, 0
        values[self.goal] = 1.
        iterations = 0
        for _ in range(c.thought_steps):
            next_values = np.zeros(c.capacity)
            next_values[self.goal] = 1.
            for (source, action, target), data in self.edges.items():
                if source == target:
                    continue
                gate = max(0., 1. - self.inhibition[source, action])
                # Reliability saturates after actual successful experience.
                gain = min(1., data['weight'] / c.connection_increment)
                drive = c.discount * gate * gain * values[target]
                next_values[source] = max(next_values[source], drive)
            iterations += 1
            if np.array_equal(next_values, values):
                break
            values = next_values
        self.goal_activity = values.copy()
        return values, iterations

    def observe(self, packet, *, actual_muscles=None, learning=True, pursue=False, propagation=True):
        packet_digest(packet)
        if actual_muscles is not None:
            actual = np.asarray(actual_muscles, float)
            if self.pending is None or actual.shape != (4,) or not np.array_equal(actual, self.pending):
                raise ValueError('Only the exact selected and executed action may be learned')
            # Validate the complete sensor outcome before committing any state.
            self.adapter._scalars(packet, actual)
            self.last_action = actual.copy()
        elif self.pending is not None:
            raise ValueError('The selected action must have an actual outcome')
        encoded = self.adapter.encode(packet, self.last_action)
        place, similarity = self._place(self._view(encoded), learning)
        cue = self._cue(packet, encoded)
        index_activity = np.zeros(self.runtime.widths['pfc'], bool)
        if place is not None:
            begin = place * self.config.assembly_size
            index_activity[begin:begin + self.config.assembly_size] = True
        index_activity[self.audio_offset:] = encoded['audio']
        timestamp = int(self.runtime.clock.当前时间)
        # One chronological record; incoming actual muscles remain aligned with
        # their true sensory outcome, retaining original memory semantics.
        self.runtime.record(encoded['visual'], encoded['audio'], self.last_action > .5,
                            index_activity, learning=learning)
        if learning and actual_muscles is not None and self.last_action.max() > .5:
            action = int(np.argmax(self.last_action))
            previous = self.previous_place
            if previous is not None:
                self.attempts[previous, action] += 1
                contact = bool(np.asarray(packet['observation']['touch']).any())
                self.inhibition[previous, action] = 1. if contact else 0.
                if place is not None and not contact:
                    key = (previous, action, place)
                    edge = self.edges.setdefault(key, dict(weight=0., source_time=self.previous_time,
                                                            motor_time=timestamp))
                    edge['weight'] += self.config.connection_increment * max(
                        0., 1. - edge['weight'] / self.config.connection_ceiling)
                    edge['source_time'], edge['motor_time'] = self.previous_time, timestamp
        if learning:
            self.learning_steps += 1
        self.previous_place, self.previous_time = place, timestamp
        values, iterations = self._propagate(propagation)
        action = None
        selected_edge = None
        candidates = []
        if pursue and place is not None and self.goal is not None:
            if place == self.goal:
                command, source = np.zeros(4), 'goal_recollection_matches_present'
            else:
                for (u, a, v), data in self.edges.items():
                    if u == place and u != v:
                        score = self.config.discount * values[v] * max(0., 1. - self.inhibition[u, a])
                        candidates.append((score, a, v, data['motor_time']))
                if candidates and max(candidates)[0] > 0:
                    score, action, successor, time = max(candidates, key=lambda item: (item[0], -item[1]))
                    command = self.runtime.recall_at_time(time, .5)['motor'].astype(float)
                    source = 'learned_relation_and_shared_time_motor'
                    selected_edge = dict(source=place, action=action, successor=successor,
                                         motor_time=time, excitation=float(score))
                else:
                    command, source = self._explore(place)
        else:
            command, source = self._explore(place)
        self.pending = command.copy()
        self.last_trace = dict(time=timestamp, clock=int(self.runtime.clock.当前时间),
            current_time_cells=int(self.runtime.clock.时间激活.sum()), place=place,
            similarity=similarity, goal=self.goal, cue=cue, propagation_steps=iterations,
            goal_activity=values[:len(self.prototypes)].tolist(), selected_edge=selected_edge,
            command=command.tolist(), source=source, learning=bool(learning),
            assemblies=len(self.prototypes), relations=len(self.edges))
        return command.copy(), copy.deepcopy(self.last_trace)

    def _explore(self, place):
        # Born curiosity samples least tried motor channels at the currently
        # recognized sensation; it does not know motor direction or geometry.
        counts = self.attempts[place].astype(float) if place is not None else np.zeros(4)
        penalties = self.inhibition[place] * 100. if place is not None else np.zeros(4)
        candidates = np.flatnonzero(counts + penalties == (counts + penalties).min())
        action = int(self.rng.choice(candidates))
        command = np.zeros(4)
        command[action] = 1.
        return command, 'born_curiosity'

    def snapshot(self):
        return dict(kind='developmental_navigation_candidate_v1', config=asdict(self.config),
            adapter=self.adapter.snapshot(), runtime=self.runtime.snapshot(),
            prototypes=self.prototypes.copy(), visits=list(self.visits),
            edges=[dict(source=u, action=a, target=v, **copy.deepcopy(data))
                   for (u, a, v), data in self.edges.items()],
            attempts=self.attempts.copy(), inhibition=self.inhibition.copy(),
            previous_place=self.previous_place, previous_time=self.previous_time,
            last_action=self.last_action.copy(), goal=self.goal, goal_time=self.goal_time,
            goal_activity=self.goal_activity.copy(), rng=self.rng.bit_generator.state,
            learning_steps=self.learning_steps, pending=copy.deepcopy(self.pending),
            last_trace=copy.deepcopy(self.last_trace))

    @classmethod
    def from_snapshot(cls, state):
        if state['kind'] != 'developmental_navigation_candidate_v1':
            raise ValueError('Wrong candidate checkpoint kind')
        obj = cls(state['config'])
        obj.adapter = SensoryAdapter.from_snapshot(state['adapter'])
        obj.runtime = OriginalRuntime.from_snapshot(state['runtime'])
        prototypes = np.asarray(state['prototypes'])
        n = len(prototypes)
        if (prototypes.dtype != np.bool_ or prototypes.shape != (n, obj.prototypes.shape[1])
                or n > obj.config.capacity or len(state['visits']) != n):
            raise ValueError('Invalid learned sensory assemblies')
        for field, shape, integral in (('attempts', (obj.config.capacity, 4), True),
                                       ('inhibition', (obj.config.capacity, 4), False),
                                       ('goal_activity', (obj.config.capacity,), False),
                                       ('last_action', (4,), False)):
            a = np.asarray(state[field])
            if a.shape != shape or not np.isfinite(a).all() or (a < 0).any():
                raise ValueError('Invalid ' + field)
            if integral and not np.issubdtype(a.dtype, np.integer):
                raise ValueError('Integer counters required')
            if not integral and (a > 1).any():
                raise ValueError('Bounded neural activity required')
        for field in ('goal', 'previous_place'):
            i = state[field]
            if i is not None and (type(i) is not int or not 0 <= i < n):
                raise ValueError('Invalid ' + field)
        for field in ('previous_time', 'goal_time'):
            t = state[field]
            if t is not None and (type(t) is not int or not 0 <= t < obj.runtime.clock.当前时间):
                raise ValueError('Invalid time reference')
        if type(state['learning_steps']) is not int or state['learning_steps'] < 0:
            raise ValueError('Invalid learning counter')
        if any(type(v) is not int or v < 0 for v in state['visits']):
            raise ValueError('Invalid sensory visit counters')
        if state['pending'] is not None:
            pending = np.asarray(state['pending'])
            if pending.shape != (4,) or not np.isfinite(pending).all() or np.any((pending < 0) | (pending > 1)):
                raise ValueError('Invalid pending muscle command')
        seen = set()
        for row in state['edges']:
            u, a, v = row['source'], row['action'], row['target']
            key = (u, a, v)
            if (any(type(i) is not int for i in key) or not 0 <= u < n or not 0 <= v < n
                    or not 0 <= a < 4 or key in seen):
                raise ValueError('Invalid or duplicate relational connection')
            seen.add(key)
            if not np.isfinite(row['weight']) or not 0 < row['weight'] <= obj.config.connection_ceiling:
                raise ValueError('Invalid relational weight')
            for name in ('source_time', 'motor_time'):
                t = row[name]
                if type(t) is not int or not 0 <= t < obj.runtime.clock.当前时间:
                    raise ValueError('Invalid relational time reference')
            muscles = obj.runtime.recall_at_time(row['motor_time'], .5)['motor']
            if int(muscles.sum()) != 1 or not muscles[a]:
                raise ValueError('Relational motor reference disagrees with shared memory')
        obj.edges = {(int(row['source']), int(row['action']), int(row['target'])):
                     {k: copy.deepcopy(v) for k, v in row.items() if k not in ('source', 'action', 'target')}
                     for row in state['edges']}
        for name in ('prototypes', 'visits', 'attempts', 'inhibition', 'previous_place', 'previous_time',
                     'last_action', 'goal', 'goal_time', 'goal_activity', 'learning_steps', 'pending', 'last_trace'):
            setattr(obj, name, copy.deepcopy(state[name]))
        obj.rng.bit_generator.state = state['rng']
        return obj
