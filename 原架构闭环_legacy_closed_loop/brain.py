"""Original sensory/PFC/time/motor classes connected to actual physical life.

Sounds have no color, rule or action labels inside this brain. The original
auditory branch of the born PFC network addresses the original hippocampal
index; that single time retrieves the co-experienced modalities. Recalled and
current signals drive the ORIGINAL recurrent excitatory/disinhibitory PFC.
There is one lifetime clock and no goal-dependent memory partition.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import copy
import hashlib
from pathlib import Path
import sys

import numpy as np

from checkpoint import save_tree, load_tree
from sensory_adapter import SensoryAdapter
from original_runtime import OriginalRuntime
from contact_reflex import contact_reflex

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from 前额叶区_prefrontal import 前额叶神经网络, 天生互惠返回线


@dataclass
class BrainConfig:
    seed: int = 2026
    sensory_threshold: float = .6
    visual_threshold: float | None = None
    audio_threshold: float | None = None
    motor_threshold: float | None = None
    audio_levels: int | None = None
    motor_levels: int | None = None
    motor_reverse_window: int | None = None
    contrast_to_silence: bool = False
    pfc_hidden_threshold: float = .55
    pfc_output_threshold: float = 1.2
    pfc_spread: int = 10
    pfc_firing_threshold: float = 2.
    pfc_hebb_increment: float = .01
    pfc_decay_floor: float = .00005
    pfc_decay_rate: float | None = None
    pfc_decay_interval: int | None = None
    pfc_decay_warning: int | None = None
    pfc_weight_ceiling: float | None = None
    pfc_weight_depression: str = 'multiplicative'
    memory_decay_rate: float | None = None
    memory_decay_interval: int | None = None
    memory_decay_warning: int | None = None
    pfc_target_high: int = 120
    pfc_feedback_inhibition: float = .12
    external_gain: float = 3.5
    recall_gain: float = 3.5
    index_threshold: float = .9
    memory_threshold: float = .5
    thought_steps: int = 2
    auditory_onset_rms: float = .02
    exploration_probability: float = .03


def packet_digest(packet):
    allowed = {'ray_distances', 'ray_colors', 'velocity', 'angular_velocity', 'pain', 'touch'}
    if set(packet) != {'observation', 'waveform'} or set(packet['observation']) != allowed:
        raise ValueError('Only actual sensor fields and PCM may enter the brain')
    h = hashlib.sha256()
    for key, value in sorted(packet['observation'].items()):
        a = np.asarray(value)
        h.update(key.encode())
        h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    h.update(np.asarray(packet.get('waveform', ()), np.float64).tobytes())
    return h.hexdigest()


class OriginalBrain:
    def __init__(self, config=None):
        self.config = config if isinstance(config, BrainConfig) else BrainConfig(**(config or {}))
        c = self.config
        self.adapter = SensoryAdapter(seed=c.seed, threshold=c.sensory_threshold,
            visual_threshold=c.visual_threshold, audio_threshold=c.audio_threshold,
            motor_threshold=c.motor_threshold, audio_levels=c.audio_levels,
            motor_levels=c.motor_levels, motor_reverse_window=c.motor_reverse_window,
            contrast_to_silence=c.contrast_to_silence)
        widths = self.adapter.widths
        self.width = sum(widths.values())
        pfc_connections = {'强化量': c.pfc_hebb_increment, '消失下限': c.pfc_decay_floor}
        memory_connections = {}
        for label, original in (('rate', '衰减率'), ('interval', '衰减间隔'), ('warning', '警戒线')):
            for prefix, parameters in (('pfc', pfc_connections), ('memory', memory_connections)):
                value = getattr(c, prefix+'_decay_'+label)
                if value is not None:
                    parameters[original] = value
        self.runtime = OriginalRuntime(widths['visual'], widths['audio'], widths['motor'], self.width,
            pfc_parameters={'门槛': c.pfc_firing_threshold, '目标下限': 4,
                            '目标上限': c.pfc_target_high, '反馈抑制': c.pfc_feedback_inhibition,
                            '强度上限': 1000.},
            pfc_connection_parameters=pfc_connections,
            memory_connection_parameters=memory_connections,
            pfc_plasticity=None if c.pfc_weight_ceiling is None else {
                'ceiling': c.pfc_weight_ceiling, 'depression': c.pfc_weight_depression})
        rng_state = np.random.get_state()
        try:
            np.random.seed(c.seed+10001)
            self.forward = 前额叶神经网络(self.width,
                隐藏层阈值=c.pfc_hidden_threshold, 输出层阈值=c.pfc_output_threshold,
                扩散宽度=c.pfc_spread)
        finally:
            np.random.set_state(rng_state)
        self.return_line = 天生互惠返回线(self.forward)
        self.rng = np.random.default_rng(c.seed+20001)
        self.thought = np.zeros(self.width, bool)
        self.last_executed = np.zeros(4)
        self.exploration_left = 0
        self.exploration_muscles = np.zeros(4)
        self.pending = None
        self.prepared = None
        self.last_info = {}
        self.flags = dict(auditory_index=True, pfc_recurrence=True, disinhibition=True,
                          recalled_drive=True, motor_recall=True, return_line=True,
                          current_drive=True, exploration=True, reflex=True)
        h = hashlib.sha256(self.adapter.birth_hash.encode())
        for arrays in (self.forward.来源, self.forward.权重, self.forward.阈值):
            for array in arrays:
                h.update(array.tobytes())
        self.birth_hash = h.hexdigest()

    def _project(self, visual, audio, motor):
        return self.forward.前向传播(visual, audio, motor).astype(bool).copy()

    def _babble(self):
        if self.exploration_left <= 0:
            # Born antagonistic muscle coordination, without goal directions.
            patterns = np.array([[0., 0., 0., 0.], [.6, 0., .6, 0.],
                [0., .5, 0., .5], [0., .3, .3, 0.], [.3, 0., 0., .3]])
            choice = int(self.rng.choice(5, p=[.03, .57, .06, .17, .17]))
            self.exploration_muscles = patterns[choice]
            self.exploration_left = int(self.rng.integers(8, 23))
        self.exploration_left -= 1
        return self.exploration_muscles.copy()

    def _reflex(self, packet):
        if not self.flags['reflex']:
            return None
        return contact_reflex(packet)

    def _microstep(self, external, *, trace):
        pfc = self.runtime.pfc
        previous = self.thought
        cancellation = pfc.去抑抵消
        excitatory_rows = pfc.联想.表
        if not self.flags['pfc_recurrence']:
            # Selective temporary lesion: local inhibitory feedback and the
            # disinhibitory projection still receive the same previous activity.
            pfc.联想.表 = {}
        if not self.flags['disinhibition']:
            pfc.去抑抵消 = 0.
        try:
            thought, info = self.runtime.pfc_microstep(previous, external, diagnostics=trace)
        finally:
            pfc.去抑抵消 = cancellation
            pfc.联想.表 = excitatory_rows
        self.thought = thought
        return info

    def _process(self, packet, *, learning, trace=False, choose_action=True):
        encoded = self.adapter.encode(packet, self.last_executed)
        v, a, m = (encoded[name] for name in ('visual', 'audio', 'motor'))
        wave = np.asarray(packet.get('waveform', ()), float)
        rms = float(np.sqrt(np.mean(wave**2))) if len(wave) else 0.
        # A general auditory input pathway, not a detector for specific tones.
        sound_cue = self._project(np.zeros_like(v), a, np.zeros_like(m))
        external = self._project(v, a, m)
        query = sound_cue if rms >= self.config.auditory_onset_rms and self.flags['auditory_index'] else self.thought
        recalled = self.runtime.recall_from_pfc(query, self.config.index_threshold, self.config.memory_threshold)
        initial_time, initial_ratio = recalled['time'], recalled['ratio']
        initial_visual = recalled['visual'].copy()
        initial_motor = recalled['motor'].copy()
        microsteps = []
        for _ in range(self.config.thought_steps):
            remembered = self._project(recalled['visual'], recalled['audio'], recalled['motor'])
            drive = self.config.external_gain*external*self.flags['current_drive']
            drive = drive + self.config.recall_gain*remembered*self.flags['recalled_drive']
            detail = self._microstep(drive, trace=trace)
            microsteps.append(detail)
            # Thought, rather than an external rule identifier, readdresses time.
            recalled = self.runtime.recall_from_pfc(self.thought,
                self.config.index_threshold, self.config.memory_threshold)
        command = np.zeros_like(m)
        if self.flags['motor_recall'] and recalled['time'] is not None:
            command = recalled['motor'].copy()
        coarse = self.return_line.返回(self.thought) if self.flags['return_line'] else np.zeros(self.width, bool)
        if not command.any() and self.flags['motor_recall']:
            command = coarse[-len(m):].copy()
        remembered_muscles = self.adapter.decode_motor(command) if command.any() else np.zeros(4)
        available = bool(command.any() and self.adapter.motor_reverse.条数() > 0)
        reflex = self._reflex(packet)
        spontaneous = choose_action and self.flags['exploration'] and (
            not available or self.exploration_left > 0 or self.rng.random() < self.config.exploration_probability)
        if not choose_action:
            muscles, source = np.zeros(4), 'terminal_observation'
        elif reflex is not None:
            muscles, source = reflex, 'body_reflex'
        elif spontaneous:
            muscles, source = self._babble(), 'spontaneous_muscles'
        elif available:
            muscles, source = remembered_muscles, 'pfc_time_motor_recall'
        else:
            muscles, source = np.zeros(4), 'rest'
        # The original index learns the born PFC forward output at this time;
        # PFC associative plasticity separately sees the actual internal thought.
        # Both refer to the same time and the same original index, not two stores.
        record = self.runtime.record(v, a, m, self.thought, learning=learning,
                                     index_pfc_bool=external)
        info = dict(frame=self.runtime.frames_recorded, index_time=record['index_time'],
            clock=int(self.runtime.clock.当前时间), time_active_count=record['time_active_count'],
            sound_rms=rms, sound_cue_cells=np.flatnonzero(sound_cue),
            external_pfc_cells=np.flatnonzero(external), thought_cells=np.flatnonzero(self.thought),
            auditory_recall_time=initial_time, auditory_recall_ratio=initial_ratio,
            initially_recalled_visual_cells=np.flatnonzero(initial_visual),
            initially_recalled_motor_cells=np.flatnonzero(initial_motor),
            final_recall_time=recalled['time'], final_recall_ratio=recalled['ratio'],
            final_visual_cells=np.flatnonzero(recalled['visual']),
            motor_command_cells=np.flatnonzero(command), remembered_muscles=remembered_muscles,
            muscles=muscles, source=source, connections=record['connections'],
            microsteps=microsteps, sensory_diagnostics=encoded['diagnostics'],
            actual_motor_in_memory=self.last_executed.copy(), learning=bool(learning))
        self.last_info = info
        return np.asarray(muscles, float), info

    def observe_then_act(self, packet, *, learning=True, trace=False):
        if self.pending is not None:
            raise RuntimeError('Confirm the actual executed action and its outcome first')
        identity = packet_digest(packet)
        self.adapter._scalars(packet, self.last_executed)
        if self.prepared is not None and self.prepared['packet_digest'] == identity:
            if self.prepared['learning'] != bool(learning):
                raise ValueError('Cannot change plasticity after processing this observation')
            muscles, info = self.prepared['muscles'].copy(), self.prepared['info']
            self.prepared = None
        else:
            # A new external sound/presentation is a new observation even if
            # the physical world has not stepped. It occupies the SAME clock.
            self.prepared = None
            muscles, info = self._process(packet, learning=learning, trace=trace)
        self.pending = {'muscles': muscles.copy(), 'learning': bool(learning)}
        return muscles.copy(), info

    def observe_outcome(self, packet, actual_muscles, *, terminal=False, trace=False):
        if self.pending is None:
            raise RuntimeError('No pending actual action')
        applied = np.asarray(actual_muscles, float)
        if applied.shape != (4,) or not np.array_equal(applied, self.pending['muscles']):
            raise ValueError('Outcome must belong to the exact selected muscles')
        if type(terminal) is not bool:
            raise ValueError('terminal must be Boolean')
        # Validate all input fields before changing learned connections.
        packet_digest(packet)
        self.adapter._scalars(packet, applied)
        learning = self.pending['learning']
        self.last_executed = applied.copy()
        if learning:
            self.adapter.learn_executed(applied)
        muscles, info = self._process(packet, learning=learning, trace=trace,
                                      choose_action=not terminal)
        self.pending = None
        self.prepared = None if terminal else dict(packet_digest=packet_digest(packet),
            muscles=muscles.copy(), info=info, learning=learning)
        return info

    def snapshot(self):
        return dict(kind='original_brain_shared_time_v1', config=asdict(self.config),
            birth_hash=self.birth_hash, adapter=self.adapter.snapshot(), runtime=self.runtime.snapshot(),
            forward={'sources': [a.copy() for a in self.forward.来源],
                     'weights': [a.copy() for a in self.forward.权重],
                     'thresholds': [a.copy() for a in self.forward.阈值],
                     'input_activity': self.forward.输入激活.copy(),
                     'layer_activity': [a.copy() for a in self.forward.普通层激活]},
            thought=self.thought.copy(), last_executed=self.last_executed.copy(),
            rng=self.rng.bit_generator.state, exploration_left=self.exploration_left,
            exploration_muscles=self.exploration_muscles.copy(), pending=self.pending,
            prepared=self.prepared, last_info=self.last_info, flags=dict(self.flags))

    @classmethod
    def from_snapshot(cls, state):
        if state['kind'] != 'original_brain_shared_time_v1':
            raise ValueError('Not a shared-time original brain checkpoint')
        obj = cls(state['config'])
        if obj.birth_hash != state['birth_hash']:
            raise ValueError('Birth identity differs')
        for name, field in (('sources', '来源'), ('weights', '权重'), ('thresholds', '阈值')):
            if not all(np.array_equal(a, b) for a, b in zip(state['forward'][name], getattr(obj.forward, field))):
                raise ValueError('Original forward connections differ')
        obj.adapter = SensoryAdapter.from_snapshot(state['adapter'])
        obj.runtime = OriginalRuntime.from_snapshot(state['runtime'])
        obj.forward.输入激活[:] = state['forward']['input_activity']
        obj.forward.普通层激活 = [a.copy() for a in state['forward']['layer_activity']]
        for name in ('thought', 'last_executed', 'exploration_muscles'):
            setattr(obj, name, np.asarray(state[name]).copy())
        for name in ('exploration_left', 'pending', 'prepared', 'last_info', 'flags'):
            setattr(obj, name, copy.deepcopy(state[name]))
        obj.rng.bit_generator.state = state['rng']
        return obj

    def save(self, path):
        save_tree(path, self.snapshot())

    @classmethod
    def load(cls, path):
        return cls.from_snapshot(load_tree(path))
