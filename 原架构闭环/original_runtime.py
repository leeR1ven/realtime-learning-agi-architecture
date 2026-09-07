"""One shared original time ring, original memories, index and PFC equations.

This is an orchestration/serialization layer. Production loading uses ordinary
imports of the original modules, including their original global initialization.
No replacement recurrent core, reward/value table, task partition or learned
sound-to-color shortcut is introduced here.

Each runtime owns one instance of the original 时间环. The original memory
classes accept old/new time explicitly when learning; their public 回忆 method
uses a module global clock. To avoid cross-instance clock mutation, this wrapper
reads those same original tables with the same voting/threshold/+1-ring rules.
The original feature->time forward offset is deliberately retained. The optional
time_ring_policy='grow' prevents future ring wrap by expanding the same clock;
the default 'fixed' preserves the original class and wrap behavior.
"""
from __future__ import annotations

import hashlib
import importlib
import inspect
from copy import deepcopy
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
SNAPSHOT_VERSION = 1
TABLE_PARAMETERS = ('强化量', '衰减率', '消失下限', '警戒线', '衰减间隔')
TABLE_COUNTERS = ('连接条数', '维护次数', '上次衰减维护')
SOURCE_FILES = ('海马体时间区.py', '视觉记忆区.py', '听觉记忆区.py',
                '运动记忆区.py', '前额叶区.py', '权重连接管理.py')


def _original_modules():
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    return {name: importlib.import_module(name) for name in
            ('海马体时间区', '视觉记忆区', '听觉记忆区', '运动记忆区', '前额叶区')}


def _bool_vector(value, width, name):
    array = np.asarray(value)
    if array.shape != (width,) or not np.isfinite(array).all() or not np.isin(array, (0, 1)).all():
        raise ValueError(f'{name} must contain {width} Boolean/0-or-1 activations')
    return array.astype(bool, copy=True)


def _configure_table(table, parameters):
    unknown = set(parameters) - set(TABLE_PARAMETERS)
    if unknown:
        raise ValueError('Unknown original connection parameters: ' + str(sorted(unknown)))
    for key, value in parameters.items():
        if not np.isfinite(value) or value < 0:
            raise ValueError('Connection parameters must be finite and nonnegative')
        if key in ('警戒线', '衰减间隔') and (int(value) != value or value < 1):
            raise ValueError('Connection count/interval parameters must be positive integers')
        if key == '强化量' and value <= 0:
            raise ValueError('Original Hebbian increment must be positive')
        if key == '衰减率' and not 0 < value <= 1:
            raise ValueError('Decay multiplier must be in (0, 1]')
        setattr(table, key, int(value) if key in ('警戒线', '衰减间隔') else float(value))


def _table_snapshot(table):
    # Preserve dictionary insertion order, including summation order on resume.
    sources = np.asarray(list(table.表), dtype=np.int64)
    offsets = [0]
    targets, weights = [], []
    for source in sources:
        row = table.表[int(source)]
        targets.extend(row)
        weights.extend(row.values())
        offsets.append(len(targets))
    return {'parameters': {key: getattr(table, key) for key in TABLE_PARAMETERS},
            'counters': {key: int(getattr(table, key)) for key in TABLE_COUNTERS},
            'sources': sources, 'offsets': np.asarray(offsets, np.int64),
            'targets': np.asarray(targets, np.int64), 'weights': np.asarray(weights, np.float64)}


def _restore_table(table, saved, source_bound, target_bound):
    sources = np.asarray(saved['sources'])
    offsets = np.asarray(saved['offsets'])
    targets = np.asarray(saved['targets'])
    weights = np.asarray(saved['weights'])
    if (sources.ndim != 1 or targets.ndim != 1 or weights.shape != targets.shape
            or offsets.shape != (len(sources)+1,) or offsets[0] != 0
            or offsets[-1] != len(targets) or (np.diff(offsets) < 0).any()):
        raise ValueError('Invalid original connection CSR shape')
    if any(not np.issubdtype(value.dtype, np.integer) for value in (sources, offsets, targets)):
        raise ValueError('Connection identifiers must be integers')
    if (len(np.unique(sources)) != len(sources) or (sources < 0).any()
            or (sources >= source_bound).any() or (targets < 0).any()
            or (targets >= target_bound).any() or not np.isfinite(weights).all()
            or (weights <= 0).any()):
        raise ValueError('Invalid original connection identifiers/weights')
    _configure_table(table, saved['parameters'])
    table.表 = {}
    for row, source in enumerate(sources):
        begin, end = int(offsets[row]), int(offsets[row+1])
        row_targets = targets[begin:end]
        if len(np.unique(row_targets)) != len(row_targets):
            raise ValueError('Duplicate original synapse')
        table.表[int(source)] = {int(target): float(weight)
                               for target, weight in zip(row_targets, weights[begin:end])}
    for key in TABLE_COUNTERS:
        setattr(table, key, int(saved['counters'][key]))
    if table.连接条数 != len(targets) or table.维护次数 < 0:
        raise ValueError('Connection counters disagree with stored edges')


class OriginalRuntime:
    """Orchestrate unchanged original class instances on one chronological ring.

    record is for the current actual modality/PFC activations, including the
    executed motor command. It learns the original frame-to-frame PFC transition
    once, and writes all three sensory/motor memories on each internal step.
    The index is written at frame-start old time, matching 前额叶区.py:408-414.
    index_pfc_bool can supply the actual same-frame fixed-network forward output
    for this index, independently of the recurrent thought being learned.
    Omitting it preserves the original wrapper's pfc_bool indexing behavior.
    No goal change or episode boundary resets the timeline or old connections.
    """

    def __init__(self, visual_width, audio_width, motor_width, pfc_width, *,
                 time_neurons=1728000, internal_step_seconds=.05, steps_per_frame=2,
                 pfc_parameters=None, pfc_connection_parameters=None,
                 memory_connection_parameters=None, pfc_plasticity=None,
                 time_ring_policy='fixed'):
        if time_ring_policy not in ('fixed', 'grow'):
            raise ValueError('Time ring policy must be fixed or grow')
        widths = dict(visual=int(visual_width), audio=int(audio_width),
                      motor=int(motor_width), pfc=int(pfc_width))
        if any(value < 1 for value in widths.values()) or int(time_neurons) < 1 or int(steps_per_frame) < 1:
            raise ValueError('Runtime widths, ring size and internal-step count must be positive')
        if not np.isfinite(internal_step_seconds) or internal_step_seconds <= 0:
            raise ValueError('Internal-step duration must be positive')
        modules = _original_modules()
        self.widths = widths
        self.clock = modules['海马体时间区'].时间环(时间神经元数量=int(time_neurons),
                       每步秒数=float(internal_step_seconds), 每帧步数=int(steps_per_frame))
        self.visual_memory = modules['视觉记忆区'].记忆区(widths['visual'])
        self.audio_memory = modules['听觉记忆区'].记忆区(widths['audio'])
        self.motor_memory = modules['运动记忆区'].记忆区(widths['motor'])
        self.memories = {'visual': self.visual_memory, 'audio': self.audio_memory, 'motor': self.motor_memory}
        # Same original time->output sets and exact vote/tie rules. The derived
        # search cache only speeds lookup and is rebuilt, never saved as memory.
        from fast_hippocampal_index import FastHippocampalIndex
        self.index = FastHippocampalIndex()
        supplied_pfc = dict(pfc_parameters or {})
        if '宽度' in supplied_pfc:
            raise ValueError('Use pfc_width to specify original PFC width')
        # Only accumulate the same currents in exact original order using
        # NumPy. Firing, inhibition adaptation and learning remain inherited.
        from fast_pfc_currents import FastPFC
        pfc_type = FastPFC
        self.pfc = pfc_type(widths['pfc'], **supplied_pfc)
        arguments = inspect.signature(pfc_type).bind(widths['pfc'], **supplied_pfc)
        arguments.apply_defaults()
        original_pfc_parameters = dict(arguments.arguments)
        original_pfc_parameters.pop('宽度')
        for memory in self.memories.values():
            for direction in ('特征到时间', '时间到特征'):
                _configure_table(getattr(memory, direction), dict(memory_connection_parameters or {}))
        for table in (self.pfc.联想, self.pfc.去抑):
            _configure_table(table, dict(pfc_connection_parameters or {}))
        self.parameters = dict(visual_width=widths['visual'], audio_width=widths['audio'],
            motor_width=widths['motor'], pfc_width=widths['pfc'], time_neurons=self.clock.时间神经元数量,
            internal_step_seconds=self.clock.每步秒数, steps_per_frame=self.clock.每帧步数,
            pfc_parameters=original_pfc_parameters,
            pfc_connection_parameters={key: getattr(self.pfc.联想, key) for key in TABLE_PARAMETERS},
            memory_connection_parameters={key: getattr(self.visual_memory.特征到时间, key) for key in TABLE_PARAMETERS})
        if time_ring_policy == 'grow':
            self.set_time_ring_policy('grow')
        if pfc_plasticity is not None:
            self.set_pfc_plasticity(pfc_plasticity)
        self.source_hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in SOURCE_FILES}
        self.previous_actual_pfc = None
        self.frames_recorded = 0
        self.internal_steps_recorded = 0
        self.pfc_microsteps = 0

    def set_time_ring_policy(self, policy):
        """Explicit capacity-policy change; keep all timestamps and memories.

        This prevents future reuse when grow is selected. It cannot undo old
        collisions if a fixed ring has already wrapped. Fixed remains the
        default, and its snapshots do not acquire an optional-policy field.
        """
        if policy not in ('fixed', 'grow'):
            raise ValueError('Time ring policy must be fixed or grow')
        current_policy = self.parameters.get('time_ring_policy', 'fixed')
        if policy == current_policy:
            return
        if policy == 'grow':
            from growing_time_ring import GrowingTimeRing
            replacement = GrowingTimeRing.from_original(self.clock)
        else:
            original_type = _original_modules()['海马体时间区'].时间环
            replacement = original_type(self.clock.时间神经元数量,
                                        self.clock.每步秒数, self.clock.每帧步数)
            replacement.当前时间 = int(self.clock.当前时间)
            replacement.时间激活[:] = self.clock.时间激活
        self.clock = replacement
        self.parameters['time_neurons'] = int(self.clock.时间神经元数量)
        if policy == 'grow':
            self.parameters['time_ring_policy'] = 'grow'
        else:
            self.parameters.pop('time_ring_policy', None)

    def set_pfc_plasticity(self, specification):
        """Explicit optional update-law change; retain all weights and counters.

        None restores additive original learning. This changes future learning,
        never the single clock, existing connection contents, or firing rule.
        """
        if specification is None:
            if 'pfc_plasticity' not in self.parameters:
                return
            from 权重连接管理 import 连接表
            for name in ('联想', '去抑'):
                old = getattr(self.pfc, name)
                table = 连接表(**{key: getattr(old, key) for key in TABLE_PARAMETERS})
                table.表 = old.表
                for key in TABLE_COUNTERS:
                    setattr(table, key, getattr(old, key))
                setattr(self.pfc, name, table)
            self.parameters.pop('pfc_plasticity', None)
            return
        if set(specification) != {'ceiling', 'depression'}:
            raise ValueError('PFC plasticity requires ceiling and depression')
        from weight_dependent_plasticity import SoftBoundConnectionTable
        replacements = {name: SoftBoundConnectionTable.from_original(
            getattr(self.pfc, name), **specification) for name in ('联想', '去抑')}
        for name, table in replacements.items():
            setattr(self.pfc, name, table)
        self.parameters['pfc_plasticity'] = dict(specification)

    def connection_counts(self):
        result = {name: {'feature_to_time': memory.特征到时间.条数(),
                         'time_to_feature': memory.时间到特征.条数()}
                  for name, memory in self.memories.items()}
        result['pfc'] = {'excitatory': self.pfc.联想.条数(), 'disinhibitory': self.pfc.去抑.条数()}
        result['index'] = self.index.连接数()
        return result

    def record(self, visual_bool, audio_bool, motor_bool, pfc_bool, *, learning=True,
               index_pfc_bool=None):
        inputs = dict(visual=_bool_vector(visual_bool, self.widths['visual'], 'visual'),
                      audio=_bool_vector(audio_bool, self.widths['audio'], 'audio'),
                      motor=_bool_vector(motor_bool, self.widths['motor'], 'motor'))
        pfc = _bool_vector(pfc_bool, self.widths['pfc'], 'pfc')
        index_pfc = (pfc if index_pfc_bool is None else
                     _bool_vector(index_pfc_bool, self.widths['pfc'], 'same-frame index PFC'))
        if self.parameters.get('time_ring_policy') == 'grow':
            # Allocation precedes both neural learning and clock mutation.
            self.clock.ensure_capacity_for_steps(self.clock.每帧步数)
            self.parameters['time_neurons'] = int(self.clock.时间神经元数量)
        had_previous = self.previous_actual_pfc is not None
        if learning and had_previous:
            self.pfc.学习(self.previous_actual_pfc, pfc)
        self.previous_actual_pfc = pfc.copy()
        steps = []
        index_time = int(self.clock.当前时间)
        for microstep in range(self.clock.每帧步数):
            old, new = self.clock.走一步()
            if learning and microstep == 0:
                self.index.学习(old, index_pfc)
            for name, memory in self.memories.items():
                if learning:
                    memory.学习(old, new, inputs[name])
                else:
                    # Frozen evaluation still has a real next sensory state;
                    # do not call 学习, since it would also advance forgetting.
                    memory.激活[:] = inputs[name]
            steps.append((int(old), int(new)))
            self.internal_steps_recorded += 1
        self.frames_recorded += 1
        return {'frame': self.frames_recorded, 'index_time': index_time, 'time_steps': steps,
                'current_time': int(self.clock.当前时间), 'time_active_count': int(self.clock.时间激活.sum()),
                'pfc_transition_learned': bool(learning and had_previous), 'learning': bool(learning),
                'index_pfc_active_count': int(index_pfc.sum()),
                'explicit_index_pfc_input': index_pfc_bool is not None,
                'active_counts': {**{name: int(value.sum()) for name, value in inputs.items()}, 'pfc': int(pfc.sum())},
                'connections': self.connection_counts()}

    def recall_at_time(self, timestamp, threshold=1.):
        if not np.isfinite(threshold) or threshold <= 0:
            raise ValueError('Recall threshold must be positive')
        if timestamp is not None and (int(timestamp) != timestamp or not 0 <= timestamp < self.clock.时间神经元数量):
            raise ValueError('Time identifier is outside the shared ring')
        result = {'time': None if timestamp is None else int(timestamp), 'threshold': float(threshold)}
        for name, memory in self.memories.items():
            current = np.zeros(memory.总数, dtype=np.float64)
            if timestamp is not None:
                for feature, weight in memory.时间到特征.查(int(timestamp)).items():
                    current[feature] += weight
            result[name+'_current'] = current
            result[name] = current >= threshold
        return result

    def recall_from_pfc(self, pfc_bool, relative_threshold=.7, threshold=1.):
        cue = _bool_vector(pfc_bool, self.widths['pfc'], 'pfc cue')
        if not np.isfinite(relative_threshold) or not 0 <= relative_threshold <= 1:
            raise ValueError('Relative index threshold must be in [0, 1]')
        timestamp, ratio = self.index.锁定时间(cue, 相对门槛=float(relative_threshold))
        result = self.recall_at_time(timestamp, threshold)
        result.update(source='pfc_index', locked=timestamp is not None, ratio=float(ratio),
                      relative_threshold=float(relative_threshold))
        return result

    def recall_from_audio(self, audio_bool, steps=1, threshold=1.):
        cue = _bool_vector(audio_bool, self.widths['audio'], 'audio cue')
        if int(steps) != steps or steps < 0:
            raise ValueError('Recall steps must be a nonnegative integer')
        # Exactly the original auditory feature->time vote, with the same
        # smallest-time argmax tie rule, represented sparsely for diagnostics.
        votes = {}
        for feature in np.flatnonzero(cue):
            for timestamp, weight in self.audio_memory.特征到时间.查(int(feature)).items():
                votes[timestamp] = votes.get(timestamp, 0.) + weight
        positive_times = sorted(timestamp for timestamp, weight in votes.items() if weight > 0)
        timestamp = max(positive_times, key=lambda item: (votes[item], -item)) if positive_times else None
        playback = []
        if timestamp is not None:
            current = timestamp
            for _ in range(int(steps)):
                if current >= self.clock.时间神经元数量:
                    # Growing chronological memory has no wrap-to-birth replay.
                    # Read-only recall does not allocate unexperienced time.
                    break
                playback.append(self.recall_at_time(current, threshold))
                current = current+1
                if self.parameters.get('time_ring_policy') != 'grow':
                    current %= self.clock.时间神经元数量
        return {'source': 'audio_feature_to_time', 'locked': timestamp is not None,
                'time': timestamp, 'peak': 0. if timestamp is None else float(votes[timestamp]),
                'voted_times': np.asarray(positive_times, np.int64),
                'voted_currents': np.asarray([votes[t] for t in positive_times], np.float64),
                'playback': playback}

    def pfc_microstep(self, previous=None, external=None, *, diagnostics=True):
        previous = self.pfc.念头.copy() if previous is None else _bool_vector(previous, self.widths['pfc'], 'previous PFC')
        if external is not None:
            external = np.asarray(external, np.float64)
            if external.shape != (self.widths['pfc'],) or not np.isfinite(external).all():
                raise ValueError('PFC external current has invalid shape or values')
        strength_before = float(self.pfc.强度)
        detail = {}
        if diagnostics:
            excitatory = self.pfc.驱动(previous)
            net_gate = self.pfc.门场(previous)
            sites = (self.widths['pfc']+self.pfc.每段-1)//self.pfc.每段
            local_count = np.bincount(np.flatnonzero(previous)//self.pfc.每段, minlength=sites)
            inhibitory = strength_before*(self.pfc.静息抑制+self.pfc.反馈抑制*local_count)
            disinhibitory = np.zeros(sites, np.float64)
            for cell in np.flatnonzero(previous):
                for site, weight in self.pfc.去抑.查(int(cell)).items():
                    disinhibitory[site] += weight
            detail = {'excitatory_current': excitatory,
                      'external_current': np.zeros(self.widths['pfc']) if external is None else external.copy(),
                      'inhibitory_site_current': inhibitory,
                      'disinhibitory_site_votes': disinhibitory,
                      'disinhibitory_site_current': self.pfc.去抑抵消*disinhibitory,
                      'net_inhibitory_site_current': net_gate,
                      'local_site_active_count': local_count}
        activity = self.pfc.微步(previous, 外部信号=external)
        self.pfc_microsteps += 1
        detail.update(current_time=int(self.clock.当前时间), pfc_microstep=self.pfc_microsteps,
            active_count=int(activity.sum()), inhibition_strength_before=strength_before,
            inhibition_strength_after=float(self.pfc.强度), threshold=float(self.pfc.门槛))
        return activity.copy(), detail

    def snapshot(self):
        parameters = deepcopy(self.parameters)
        parameters['time_neurons'] = int(self.clock.时间神经元数量)
        # Preserve explicit runtime parameter interventions as well as birth
        # values; the original class keeps these as public attributes.
        for key in parameters['pfc_parameters']:
            if hasattr(self.pfc, key):
                parameters['pfc_parameters'][key] = getattr(self.pfc, key)
        memories = {}
        for name, memory in self.memories.items():
            memories[name] = {'activity': memory.激活.copy(),
                              'feature_to_time': _table_snapshot(memory.特征到时间),
                              'time_to_feature': _table_snapshot(memory.时间到特征)}
        times = list(self.index.时间到输出)
        offsets, features = [0], []
        for timestamp in times:
            features.extend(sorted(self.index.时间到输出[timestamp]))
            offsets.append(len(features))
        return {'snapshot_version': SNAPSHOT_VERSION, 'parameters': parameters,
            'original_source_sha256': dict(self.source_hashes),
            'clock': {'current_time': int(self.clock.当前时间), 'activity': self.clock.时间激活.copy()},
            'memories': memories,
            'index': {'times': np.asarray(times, np.int64), 'offsets': np.asarray(offsets, np.int64),
                      'features': np.asarray(features, np.int64)},
            'pfc': {'activity': self.pfc.念头.copy(), 'inhibition_strength': float(self.pfc.强度),
                    'excitatory': _table_snapshot(self.pfc.联想), 'disinhibitory': _table_snapshot(self.pfc.去抑)},
            'previous_actual_pfc': None if self.previous_actual_pfc is None else self.previous_actual_pfc.copy(),
            'frames_recorded': self.frames_recorded, 'internal_steps_recorded': self.internal_steps_recorded,
            'pfc_microsteps': self.pfc_microsteps}

    @classmethod
    def from_snapshot(cls, saved):
        if saved.get('snapshot_version') != SNAPSHOT_VERSION:
            raise ValueError('Unsupported original runtime snapshot')
        obj = cls(**saved['parameters'])
        if saved['original_source_sha256'] != obj.source_hashes:
            raise ValueError('Original source files differ from the runtime snapshot')
        n_time = obj.clock.时间神经元数量
        current = int(saved['clock']['current_time'])
        activity = _bool_vector(saved['clock']['activity'], n_time, 'shared clock')
        if not 0 <= current < n_time or activity.sum() != 1 or not activity[current]:
            raise ValueError('Shared clock must have exactly its current time active')
        obj.clock.当前时间 = current
        obj.clock.时间激活[:] = activity
        for name, memory in obj.memories.items():
            item = saved['memories'][name]
            memory.激活[:] = _bool_vector(item['activity'], memory.总数, name)
            _restore_table(memory.特征到时间, item['feature_to_time'], memory.总数, n_time)
            _restore_table(memory.时间到特征, item['time_to_feature'], n_time, memory.总数)
        index = saved['index']
        times, offsets, features = (np.asarray(index[key]) for key in ('times', 'offsets', 'features'))
        if (times.ndim != 1 or features.ndim != 1 or offsets.shape != (len(times)+1,)
                or offsets[0] != 0 or offsets[-1] != len(features) or (np.diff(offsets) < 0).any()
                or (times < 0).any() or (times >= n_time).any() or len(np.unique(times)) != len(times)
                or (features < 0).any() or (features >= obj.widths['pfc']).any()
                or any(not np.issubdtype(a.dtype, np.integer) for a in (times, offsets, features))):
            raise ValueError('Invalid original hippocampal index')
        obj.index.时间到输出 = {}
        for row, timestamp in enumerate(times):
            obj.index.时间到输出[int(timestamp)] = set(map(int, features[int(offsets[row]):int(offsets[row+1])]))
        obj.index.rebuild_from_original()
        pfc = saved['pfc']
        obj.pfc.念头[:] = _bool_vector(pfc['activity'], obj.widths['pfc'], 'PFC state')
        if not np.isfinite(pfc['inhibition_strength']):
            raise ValueError('Invalid original PFC inhibition strength')
        obj.pfc.强度 = float(pfc['inhibition_strength'])
        _restore_table(obj.pfc.联想, pfc['excitatory'], obj.widths['pfc'], obj.widths['pfc'])
        _restore_table(obj.pfc.去抑, pfc['disinhibitory'], obj.widths['pfc'],
                       (obj.widths['pfc']+obj.pfc.每段-1)//obj.pfc.每段)
        obj.previous_actual_pfc = (None if saved['previous_actual_pfc'] is None else
                                  _bool_vector(saved['previous_actual_pfc'], obj.widths['pfc'], 'previous actual PFC'))
        for key in ('frames_recorded', 'internal_steps_recorded', 'pfc_microsteps'):
            value = saved[key]
            if int(value) != value or value < 0:
                raise ValueError('Invalid runtime counter')
            setattr(obj, key, int(value))
        return obj


Runtime = OriginalRuntime
