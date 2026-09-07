"""Transparent integration of sensory facts, Hebbian PFC and muscle competition.

The public boundary has two steps: decide from sensors, then observe the actual
physical outcome. Never pass route labels, world coordinates or an action oracle.
Original navigation baselines and checkpoints remain separate.
"""
from __future__ import annotations

from collections import deque
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/'色标导航_color_beacon_navigation'))
from associative_controller import Controller as EventController, MOTOR_TEMPLATES
from route_switch_controller import RouteSwitchController, _five_frequency_receptors
from recurrent_core import SparseHebbianPFC

KIND = 'integrated_factual_recurrent_v1'


def _pack_tree(value, arrays):
    if isinstance(value, np.ndarray):
        key = 'integrated_array_' + str(len(arrays))
        arrays[key] = value
        return {'array_key': key}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _pack_tree(item, arrays) for key, item in value.items()}
    if isinstance(value, (list, tuple, deque)):
        return [_pack_tree(item, arrays) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(type(value))


def _unpack_tree(value, arrays):
    if isinstance(value, dict):
        if set(value) == {'array_key'}:
            return arrays[value['array_key']].copy()
        return {key: _unpack_tree(item, arrays) for key, item in value.items()}
    if isinstance(value, list):
        return [_unpack_tree(item, arrays) for item in value]
    return value


class IntegratedController(RouteSwitchController):
    def __init__(self, *, mixed_neurons=8192, max_events=100000,
                 receptor_levels=64, **kwargs):
        super().__init__(mixed_neurons=mixed_neurons, max_events=max_events,
                         receptor_levels=receptor_levels, **kwargs)
        # All original four touch receptors survive; metric distance covers 16m.
        self.sensory_channels = 132
        self.receptor_count = self.sensory_channels*self.receptor_levels*2
        self.view_sources = self.rng.integers(0, self.receptor_count,
                                             (self.mixed_neurons, 2), dtype=np.int32)
        self.shared_sensory = np.zeros(self.mixed_neurons, dtype=bool)
        self.shared_sensory[self.rng.permutation(self.mixed_neurons)[:self.mixed_neurons//8]] = True
        self.event_views = np.zeros((self.max_events, self.sensory_channels), np.float32)
        self.event_outcomes = np.zeros_like(self.event_views)
        self.event_terminal = np.zeros(self.max_events, bool)
        self.fact_cache = [None]*self.mixed_neurons
        self.core = SparseHebbianPFC(width=self.mixed_neurons)
        self.core_enabled = True
        self.motor_competition_enabled = True
        self.prediction_inhibition_enabled = True
        self.motor_error_limit = .065
        self.error_scale = .07
        self.thought_gain = .35
        self.fact_capacity_retirements = 0
        self.outcomes_observed = 0
        self.last_cognitive_info = {}
        self.birth_hash = self._birth_hash()
        self.reset_activity()

    def _birth_hash(self):
        base = super()._birth_hash()
        if not hasattr(self, 'shared_sensory'):
            return base
        h = hashlib.sha256((KIND+base+'linear-distance/16,touch4').encode())
        h.update(self.shared_sensory.tobytes())
        return h.hexdigest()

    def reset_activity(self):
        super().reset_activity()
        self.pending = None
        self.previous_values = None
        self.predicted_outcome = None
        self.imagined_outcome = None
        self.recalled_current = np.zeros(self.mixed_neurons, np.float32)
        self.last_prediction_error = 0.
        self.last_cognitive_info = {}
        if hasattr(self, 'core'):
            self.core.reset_activity()

    def _observe(self, packet):
        # Retain the original strict finite/range validation, then preserve the
        # original four touch channels and a fixed metric distance encoding.
        super()._observe(packet)
        o = packet['observation']
        touch = np.asarray(o.get('touch', (0.,)*4), np.float32)
        if touch.shape != (4,) or not np.isfinite(touch).all():
            raise ValueError('Four actual touch receptors are required')
        values = np.concatenate((np.clip(np.asarray(o['ray_distances'])/16., 0., 1.),
            np.asarray(o['ray_colors']).ravel(),
            np.clip((np.asarray(o['velocity'])+3.)/6., 0., 1.),
            [np.clip((o['angular_velocity']+4.)/8., 0., 1.), np.clip(o.get('pain', 0.), 0., 1.)],
            np.clip(touch, 0., 1.))).astype(np.float32)
        return values, _five_frequency_receptors(packet.get('waveform', ()))

    def _update_context(self, audio):
        # Fixed acoustic competition, without erasing a remembered phase on a
        # new rule. Later current and prediction error can inhibit stale recall.
        old_goal = self._goal_signature()
        g = audio[:3]
        self.context[:3] = (.85*self.context[:3]+1.7*g-.9*g.max(initial=0.) >= .5)
        self.context[3] = 0.
        r = np.r_[0., audio[3:5]]
        current = .85*self.rule_context+1.9*r-1.15*r.max(initial=0.)
        current[0] += .55*max(0., 1.-float(self.rule_context[1:].sum()))
        self.rule_context = (current >= .5).astype(np.float32)
        if old_goal and self._goal_signature() != old_goal:
            self.recent_actions.clear()
            self.previous_event_id = -1
            self.last_recalled_ids = np.empty(0, np.int64)
            self.last_recalled_weights = np.empty(0, np.float32)
            self.recalled_current[:] = 0.

    def _mixed_activity(self, values):
        positive = values[:, None] >= self.levels
        receptors = np.stack((positive, ~positive), axis=-1).ravel()
        if not self.view_enabled:
            receptors[:] = False
        sensory = receptors[self.view_sources].sum(axis=1)
        goal = self.context[self.context_source] if self.goal_enabled and self.context_enabled else 1.
        rule = self.rule_context[self.rule_source] if self.rule_enabled else 1.
        active = np.where(self.shared_sensory, sensory >= 2, sensory+goal+rule >= 3.5)
        return receptors, np.flatnonzero(active).astype(np.int32)

    def _votes(self, active):
        scores = np.zeros(self.max_events, np.float32)
        for feature in active:
            feature = int(feature)
            row = self.feature_to_events[feature]
            if row:
                slots = self.fact_cache[feature]
                if slots is None:
                    slots = np.fromiter(row, np.int32, count=len(row))
                    self.fact_cache[feature] = slots
                scores[slots] += 1./np.sqrt(len(row))
        scores /= np.sqrt(self.event_active_count)
        return scores

    @staticmethod
    def _attention(scores, cap=384):
        slots = np.flatnonzero(scores > 0)
        if len(slots) > cap:
            slots = slots[np.argpartition(scores[slots], -cap)[-cap:]]
        return slots

    @staticmethod
    def _error(memory, actual):
        d = np.asarray(memory)-actual
        return np.sqrt(.45*np.mean(d[..., :31]**2, axis=-1)
            + .45*np.mean(d[..., 31:124]**2, axis=-1)
            + .10*np.mean(d[..., 124:]**2, axis=-1))

    def _cognitive_readout(self, values, external, thought):
        evidence = self._votes(external)
        imagined = self._votes(np.flatnonzero(thought)) if self.core_enabled else np.zeros_like(evidence)
        peak = float(evidence.max(initial=0.))
        imagined_peak = float(imagined.max(initial=0.))
        scores = evidence.copy()
        if imagined_peak > 0 and peak > 0:
            scores += self.thought_gain*peak*imagined/imagined_peak
        # Shared sensory facts may be recalled across rules. Motor value is
        # evaluated under current neural goal/rule identity, not a route label.
        signature = self._context_signature()
        compatible = self.event_context == signature
        motor_seed = scores*self.event_strength*compatible
        candidates = np.union1d(self._attention(scores), self._attention(motor_seed)).astype(np.int32)
        if not len(candidates):
            return np.zeros(4, np.float32), False, {
                'fact_recall_events': [], 'recall_events': [], 'recall_weights': [],
                'motor_error': None, 'source': 'unfamiliar_exploration'}
        errors = self._error(self.event_views[candidates], values)
        # Opponent sensory mismatch inhibits an event even when all alternatives
        # are poor. No relative-to-best visual pass grants false familiarity.
        agreement = np.exp(-(errors/self.error_scale)**2)
        fit = scores[candidates]*agreement if self.prediction_inhibition_enabled else scores[candidates].copy()
        top = np.argsort(fit)[-self.max_recalled_events:]
        fact_slots = candidates[top]
        fact_weights = fit[top]/max(float(fit[top].sum()), 1e-12)
        # Outcomes are real post-action observations, not invented future facts.
        self.imagined_outcome = fact_weights @ self.event_outcomes[fact_slots]
        self._prediction_slots = fact_slots
        self._prediction_weights = fact_weights
        _, future = self._mixed_activity(self.imagined_outcome)
        self.recalled_current[:] = 0.
        self.recalled_current[future] = .65

        sequence = np.zeros(len(candidates), np.float32)
        index = {int(slot): i for i, slot in enumerate(candidates)}
        if self.sequence_enabled:
            for event_id, weight in zip(self.last_recalled_ids, self.last_recalled_weights):
                old = self.event_slot_by_id.get(int(event_id))
                if old is not None:
                    successor = self.event_slot_by_id.get(int(self.event_next[old]))
                    if successor in index:
                        sequence[index[successor]] += weight
        # Predictive sequence cannot bypass absolute reality mismatch at action
        # readout. Internal PFC activation itself is not clipped to current view.
        motor = (fit + .18*max(peak, 1e-8)*sequence*agreement)*self.event_strength[candidates]*compatible[candidates]
        valid = (motor > 0) & (errors <= self.motor_error_limit)
        if not self.motor_memory_enabled:
            valid[:] = False
        selected = np.flatnonzero(valid)
        available = bool(len(selected))
        before_current = np.zeros(4)
        winner = None
        if available:
            selected = selected[np.argsort(motor[selected])[-self.max_recalled_events:]]
            selected = selected[motor[selected] >= .75*motor[selected].max()]
            initial_weights = motor[selected]/motor[selected].sum()
            before_current = initial_weights @ self.event_muscles[candidates[selected]]
            if self.motor_competition_enabled:
                motions = self.event_muscles[candidates[selected]]
                groups = np.argmin(((motions[:, None]-MOTOR_TEMPLATES[None])**2).sum(axis=2), axis=1)
                group_current = np.bincount(groups, weights=motor[selected], minlength=len(MOTOR_TEMPLATES))
                winner = int(np.argmax(group_current))
                selected = selected[groups == winner]
            weights = motor[selected]/max(float(motor[selected].sum()), 1e-12)
            slots = candidates[selected]
            current = weights @ self.event_muscles[slots]
            self.last_recalled_ids = self.event_ids[slots].copy()
            self.last_recalled_weights = weights.astype(np.float32)
        else:
            current, slots, weights = np.zeros(4), np.empty(0, np.int32), np.empty(0)
            self.last_recalled_ids = self.event_ids[fact_slots].copy()
            self.last_recalled_weights = fact_weights.astype(np.float32)
        trusted = (self.event_strength[candidates] > 0) & compatible[candidates]
        info = {'fact_recall_events': self.event_ids[fact_slots].tolist(),
            'fact_recall_unrewarded': int(np.count_nonzero(self.event_strength[fact_slots] <= 0)),
            'recall_events': self.event_ids[slots].tolist(), 'recall_weights': weights.tolist(),
            'motor_error': float(errors[trusted].min()) if trusted.any() else None,
            'minimum_fact_error': float(errors.min()), 'motor_group': winner,
            'muscles_before_competition': before_current.tolist(),
            'sequence_supported_events': self.event_ids[candidates[sequence > 0]].tolist()}
        return np.asarray(current, np.float32), available, info

    def _physical_reflex(self, packet):
        # The same innate short-range/contact withdrawal, with raw receptors;
        # changing metric encoding must not invert the old proximity reflex.
        o = packet['observation']
        distances = np.asarray(o['ray_distances'])
        proximity = np.exp(-distances/2.)
        old_values = np.zeros(129, np.float32)
        old_values[:31] = proximity
        old_values[-2] = float(o.get('pain', 0.))
        old_values[-1] = float(np.max(o['touch']))
        return super()._body_reflex(old_values)

    def observe_then_act(self, packet, teacher_muscles=None, learning=True):
        if teacher_muscles is not None:
            raise ValueError('This integrated experiment uses autonomous actions only')
        if self.pending is not None:
            raise RuntimeError('Observe the real outcome before deciding another action')
        values, audio = self._observe(packet)
        self._prediction_slots = np.empty(0, np.int32)
        self._prediction_weights = np.empty(0, np.float32)
        self._update_context(audio)
        receptors, external = self._mixed_activity(values)
        external_current = np.zeros(self.mixed_neurons, np.float32)
        external_current[external] = 1.6
        if self.core_enabled:
            thought, core_info = self.core.advance(external_current, self.recalled_current, learning=False)
        else:
            thought, core_info = external_current > 0, {}
        self.pfc_activity = np.asarray(thought, bool)
        memory, available, info = self._cognitive_readout(values, external, self.pfc_activity)
        reflex = self._physical_reflex(packet)
        if available and self.exploration and self.exploration_burst <= 0 and self.rng.random() < self.exploration_rate:
            self.exploration_burst = int(self.rng.integers(5, 10))
            self.exploration_left = 0
        if reflex is not None:
            muscles, source = reflex, 'body_reflex'
        elif self.exploration_burst > 0:
            self.exploration_burst -= 1
            muscles, source = self._explore(), 'persistent_exploration'
        elif available:
            muscles, source = memory, 'recurrent_associative_memory'
        else:
            muscles, source = self._explore(), 'unfamiliar_exploration'
        muscles = np.clip(muscles, 0., 1.).astype(np.float32)
        # A prediction of an executed consequence must condition on that action.
        # Unconditional imagined alternatives remain a separate PFC current.
        self.predicted_outcome = None
        if len(self._prediction_slots):
            action_error = np.linalg.norm(self.event_muscles[self._prediction_slots]-muscles, axis=1)
            match = action_error <= .25
            if match.any():
                weight = self._prediction_weights[match]*np.exp(-(action_error[match]/.15)**2)
                weight /= max(float(weight.sum()), 1e-12)
                self.predicted_outcome = weight @ self.event_outcomes[self._prediction_slots[match]]
        self.pending = {'values': values.copy(), 'features': external.copy(),
            'muscles': muscles.copy(), 'signature': self._context_signature(), 'learning': bool(learning)}
        info.update(frame=self.frame, source=source, predicted_muscles=muscles.tolist(),
            muscle_currents=memory.tolist(), teaching=False, goal_active=np.flatnonzero(self.context[:3]).tolist(),
            rule_active=np.flatnonzero(self.rule_context).tolist(), context_signature=self._context_signature(),
            sensory_active=int(receptors.sum()), pfc_active_count=int(self.pfc_activity.sum()),
            external_pfc_count=len(external), core_enabled=self.core_enabled,
            core=core_info, last_prediction_error=self.last_prediction_error,
            memory_events=len(self.event_slot_by_id), trusted_events=int(np.count_nonzero(self.event_strength > 0)))
        self.last_info = info
        return muscles.copy(), info

    def _remember(self, active, values, actual_muscles, taught=False):
        # Bounded hippocampal time ring. Facts keep arriving even when all older
        # events have value. Retirement removes incident edges; no false bridge.
        slot = self.next_slot
        old_id = int(self.event_ids[slot])
        if old_id >= 0:
            previous = self.event_slot_by_id.get(int(self.event_previous[slot]))
            following = self.event_slot_by_id.get(int(self.event_next[slot]))
            if previous is not None:
                self.event_next[previous] = -1
            if following is not None:
                self.event_previous[following] = -1
            self.event_slot_by_id.pop(old_id, None)
            for feature in self.event_features[slot]:
                feature = int(feature)
                self.feature_to_events[feature].discard(slot)
                self.readout_feature_to_events[feature].discard(slot)
                self.readout_cache[feature] = None
                self.fact_cache[feature] = None
            self.fact_capacity_retirements += 1
        self.event_features[slot] = active.copy()
        for feature in active:
            feature = int(feature)
            self.feature_to_events[feature].add(slot)
            self.fact_cache[feature] = None
        self.event_views[slot] = values
        self.event_muscles[slot] = actual_muscles
        self.event_strength[slot] = 0.
        self.event_kind[slot] = 1
        self.event_ids[slot] = self.next_event_id
        previous_id = self.previous_event_id if self.previous_event_id in self.event_slot_by_id else -1
        self.event_previous[slot] = previous_id
        self.event_next[slot] = -1
        self.event_context[slot] = self.pending['signature']
        self.event_frame[slot] = self.frame
        self.event_active_count[slot] = max(1, len(active))
        if previous_id >= 0:
            self.event_next[self.event_slot_by_id[previous_id]] = self.next_event_id
        self.event_slot_by_id[self.next_event_id] = slot
        self.recent_actions.append((slot, self.next_event_id))
        self.previous_event_id = self.next_event_id
        self.next_event_id += 1
        self.next_slot = (slot+1) % self.max_events
        return slot

    def observe_outcome(self, packet, actual_muscles, *, reward=0., terminal=False, learning=None):
        if self.pending is None:
            raise RuntimeError('No unobserved physical action')
        pending = self.pending
        applied = np.asarray(actual_muscles, np.float32)
        if applied.shape != (4,) or not np.array_equal(applied, pending['muscles']):
            raise ValueError('Outcome must correspond to the actually selected four muscles')
        if learning is None:
            learning = pending['learning']
        if bool(learning) != pending['learning']:
            raise ValueError('Decision and outcome must use the same learning mode')
        if not np.isfinite(reward) or reward < 0 or reward > 1:
            raise ValueError('Reward must be a finite scalar in [0,1]')
        outcome, audio = self._observe(packet)
        slot = None
        if learning:
            slot = self._remember(pending['features'], pending['values'], applied)
            self.event_outcomes[slot] = outcome
            self.event_terminal[slot] = bool(terminal)
        self._update_context(audio)
        _, after = self._mixed_activity(outcome)
        if learning and self.core_enabled:
            self.core.observe_transition(self.core.activity_from_indices(pending['features']),
                                         self.core.activity_from_indices(after), gain=1.)
        if self.predicted_outcome is not None:
            self.last_prediction_error = float(self._error(self.predicted_outcome, outcome))
            if self.prediction_inhibition_enabled:
                self.recalled_current *= np.exp(-(self.last_prediction_error/self.error_scale)**2)
        if learning and reward > 0:
            self.receive_reward(float(reward))
        self.previous_values = pending['values'].copy()
        self.pending = None
        self.frame += 1
        self.outcomes_observed += 1
        return {'recorded_event': None if slot is None else int(self.event_ids[slot]),
                'terminal_sensation_observed': bool(terminal),
                'prediction_error': self.last_prediction_error,
                'reward': float(reward), 'learning': bool(learning)}

    def save(self, path):
        destination = Path(path).resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        fd, base_path = tempfile.mkstemp(dir=destination.parent, suffix='.npz')
        os.close(fd)
        try:
            EventController.save(self, base_path)
            with np.load(base_path, allow_pickle=False) as data:
                arrays = {key: data[key].copy() for key in data.files}
            metadata = json.loads(arrays['metadata'].tobytes().decode())
            extras = {name: getattr(self, name) for name in (
                'rule_source', 'rule_context', 'shared_sensory', 'event_outcomes', 'event_terminal',
                'core_enabled', 'goal_enabled', 'rule_enabled', 'rule_interventions',
                'motor_competition_enabled', 'prediction_inhibition_enabled', 'motor_error_limit',
                'error_scale', 'thought_gain', 'fact_capacity_retirements', 'outcomes_observed',
                'pending', 'previous_values', 'predicted_outcome', 'imagined_outcome',
                'recalled_current', 'last_prediction_error')}
            extras['core'] = self.core.snapshot()
            metadata.update(controller_kind=KIND, integrated_state=_pack_tree(extras, arrays))
            arrays['metadata'] = np.frombuffer(json.dumps(metadata, ensure_ascii=False).encode(), np.uint8)
            fd, out = tempfile.mkstemp(dir=destination.parent, suffix='.tmp')
            try:
                with os.fdopen(fd, 'wb') as stream:
                    np.savez_compressed(stream, **arrays)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(out, destination)
            finally:
                Path(out).unlink(missing_ok=True)
        finally:
            Path(base_path).unlink(missing_ok=True)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(data['metadata'].tobytes().decode())
            if meta.get('controller_kind') != KIND:
                raise ValueError('Integrated sensory/PFC identity requires its own checkpoint')
            extra = _unpack_tree(meta['integrated_state'], data)
        obj = EventController.load.__func__(cls, path)
        core = extra.pop('core')
        for name, value in extra.items():
            setattr(obj, name, value)
        obj.core = SparseHebbianPFC.from_snapshot(core)
        return obj


Controller = IntegratedController
