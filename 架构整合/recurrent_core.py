"""Standalone temporal-Hebbian PFC: recurrence, inhibition and disinhibition.

Semantics follow the original 前额叶联想区 without importing its module-level
randomly born networks. Real observed cell transitions strengthen excitatory
cell->cell and cell->inhibitory-site disinhibitory synapses. Internal replay is
a separate current source, never silently written as an observed experience.

Numerical changes are explicit: finite synaptic efficacy, division by total
presynaptic activity, and site activity expressed as a density. This avoids the
old raw sum increasing 200-fold merely because 200 cells represent a pattern.
Storage allocates a float32 row only when that source acquires connections.
All nonzero connections are exported; no top-K activity or connection pruning.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np


SNAPSHOT_VERSION = 1


@dataclass(frozen=True)
class CoreConfig:
    width: int = 8192
    site_width: int = 40
    threshold: float = .18
    hebb_rate: float = .25
    synapse_ceiling: float = 1.
    external_gain: float = 1.
    recalled_gain: float = .6
    recurrent_gain: float = 1.
    disinhibition_gain: float = 1.3
    resting_inhibition: float = .30
    local_feedback_gain: float = .15
    membrane_decay: float = .20
    adaptation_decay: float = .80
    adaptation_gain: float = .05
    initial_inhibition: float = 1.
    minimum_inhibition: float = .5
    maximum_inhibition: float = 8.
    target_low_fraction: float = .008
    target_high_fraction: float = .06
    homeostasis_rate: float = .10
    decay_interval: int = 200
    decay_factor: float = .995
    disappearance_threshold: float = .0001
    prune_rows_per_tick: int = 16


class _LazyRows:
    """Lazy dense rows, complete sparse CSR snapshots, uniform lazy decay.

    Stored values are effective weights / global_scale. Uniform decay changes
    that one shared scale. Periodic scanning may erase only weights that have
    fallen below the same disappearance threshold. A row learned since the
    most recent uniform decay is protected until the next uniform decay.
    """

    def __init__(self, sources, targets):
        self.sources, self.targets = int(sources), int(targets)
        self.rows = {}
        self.row_counts = np.zeros(self.sources, dtype=np.int64)
        self.last_learning_decay = np.zeros(self.sources, dtype=np.int64)
        self.global_scale = 1.
        self.decay_count = 0
        self.prune_cursor = 0
        self.edge_count = 0
        self.removed_by_decay = 0

    def learn(self, pre_ids, pre_values, post_ids, post_values, rate, ceiling):
        if not len(pre_ids) or not len(post_ids) or rate <= 0:
            return
        for source, amplitude in zip(pre_ids, pre_values):
            source = int(source)
            row = self.rows.get(source)
            if row is None:
                row = np.zeros(self.targets, dtype=np.float32)
                self.rows[source] = row
            old = row[post_ids]
            added = int(np.count_nonzero(old == 0))
            effective = old * self.global_scale
            step = np.minimum(1., rate * float(amplitude) * post_values)
            updated = effective + step * (ceiling - effective)
            row[post_ids] = np.minimum(ceiling, updated) / self.global_scale
            self.row_counts[source] += added
            self.edge_count += added
            self.last_learning_decay[source] = self.decay_count

    def propagate(self, activity):
        active = np.flatnonzero(activity > 0)
        denominator = max(float(np.sum(activity, dtype=np.float64)), 1.)
        result = np.zeros(self.targets, dtype=np.float32)
        used_edges = 0
        # One vector operation per active source, not a Python loop per synapse.
        for source in active:
            row = self.rows.get(int(source))
            if row is not None:
                result += row * float(activity[source])
                used_edges += int(self.row_counts[source])
        result *= self.global_scale / denominator
        return result, used_edges

    def decay(self, factor):
        self.global_scale *= float(factor)
        self.decay_count += 1
        if self.global_scale < 1e-8:
            # Rare exact rescaling prevents representational under/overflow;
            # it changes no effective synapse and discards no connection.
            for row in self.rows.values():
                row *= self.global_scale
            self.global_scale = 1.

    def prune_decayed_rows(self, number, threshold):
        if not self.decay_count or threshold <= 0:
            return 0
        removed = 0
        for _ in range(min(int(number), self.sources)):
            source = self.prune_cursor
            self.prune_cursor = (source + 1) % self.sources
            row = self.rows.get(source)
            if row is None or self.last_learning_decay[source] >= self.decay_count:
                continue
            small = (row > 0) & (row < threshold / self.global_scale)
            count = int(np.count_nonzero(small))
            if count:
                row[small] = 0.
                self.row_counts[source] -= count
                self.edge_count -= count
                removed += count
                if not self.row_counts[source]:
                    del self.rows[source]
        self.removed_by_decay += removed
        return removed

    @property
    def allocated_bytes(self):
        return len(self.rows) * self.targets * np.dtype(np.float32).itemsize

    def export(self, prefix):
        counts = self.row_counts.copy()
        indptr = np.concatenate(([0], np.cumsum(counts, dtype=np.int64)))
        indices = np.empty(int(indptr[-1]), dtype=np.int32)
        weights = np.empty(int(indptr[-1]), dtype=np.float32)
        materialized = np.zeros(self.sources, dtype=bool)
        for source, row in self.rows.items():
            targets = np.flatnonzero(row)
            assert len(targets) == counts[source]
            begin, end = int(indptr[source]), int(indptr[source + 1])
            indices[begin:end] = targets
            weights[begin:end] = row[targets]
            materialized[source] = True
        arrays = {prefix + '_indptr': indptr, prefix + '_indices': indices,
                  prefix + '_raw_weights': weights, prefix + '_materialized': materialized,
                  prefix + '_last_learning_decay': self.last_learning_decay.copy()}
        state = dict(global_scale=self.global_scale, decay_count=self.decay_count,
                     prune_cursor=self.prune_cursor, edge_count=self.edge_count,
                     removed_by_decay=self.removed_by_decay)
        return arrays, state

    def restore(self, prefix, arrays, state):
        indptr = np.asarray(arrays[prefix + '_indptr'])
        indices = np.asarray(arrays[prefix + '_indices'])
        weights = np.asarray(arrays[prefix + '_raw_weights'])
        materialized = np.asarray(arrays[prefix + '_materialized'])
        epochs = np.asarray(arrays[prefix + '_last_learning_decay'])
        if indptr.shape != (self.sources + 1,) or indptr[0] != 0 or (np.diff(indptr) < 0).any():
            raise ValueError('Invalid CSR row offsets')
        if indptr[-1] != len(indices) or weights.shape != indices.shape:
            raise ValueError('Invalid CSR edge counts')
        if (indices < 0).any() or (indices >= self.targets).any() or (weights <= 0).any() or not np.isfinite(weights).all():
            raise ValueError('Invalid CSR targets or weights')
        if materialized.shape != (self.sources,) or epochs.shape != (self.sources,):
            raise ValueError('Invalid sparse row state')
        self.rows = {}
        self.row_counts = np.diff(indptr).astype(np.int64)
        for source in np.flatnonzero(materialized | (self.row_counts > 0)):
            begin, end = int(indptr[source]), int(indptr[source + 1])
            targets = indices[begin:end]
            if len(targets) and (np.diff(targets) <= 0).any():
                raise ValueError('CSR targets must be unique and ordered')
            row = np.zeros(self.targets, dtype=np.float32)
            row[targets] = weights[begin:end]
            self.rows[int(source)] = row
        self.last_learning_decay = epochs.astype(np.int64, copy=True)
        for key in ('global_scale', 'decay_count', 'prune_cursor', 'edge_count', 'removed_by_decay'):
            setattr(self, key, state[key])
        if not np.isfinite(self.global_scale) or self.global_scale <= 0 or self.edge_count != len(indices):
            raise ValueError('Invalid synaptic scale/count state')


class SparseHebbianPFC:
    """Persistent recurrent PFC with explicit observed-transition plasticity.

    advance(external_current, recalled_current=None, learning=False) returns
    (boolean activity, diagnostics). It never learns its own generated thoughts.
    learning=True advances only the uniform maintenance clock; normal embodied
    use should leave it False and call observe_transition after real physics.
    observe_transition(..., gain=1) also advances that maintenance clock once.

    snapshot() returns {'metadata': JSON-safe dict, 'arrays': numeric ndarrays}.
    A caller can place metadata JSON and every array in an allow_pickle=False NPZ.
    """

    def __init__(self, width=8192, site_width=40, **parameters):
        self.config = CoreConfig(width=int(width), site_width=int(site_width), **parameters)
        c = self.config
        if c.width < 1 or c.site_width < 1 or c.decay_interval < 1 or c.prune_rows_per_tick < 1:
            raise ValueError('Widths, decay interval and scan budget must be positive')
        numeric = [v for v in asdict(c).values() if isinstance(v, (int, float))]
        if not np.isfinite(numeric).all() or any(v < 0 for v in numeric):
            raise ValueError('Core parameters must be finite and nonnegative')
        if not 0 <= c.membrane_decay < 1 or not 0 <= c.adaptation_decay < 1 or not 0 < c.decay_factor <= 1:
            raise ValueError('Invalid decay parameter')
        if c.synapse_ceiling <= 0 or c.hebb_rate <= 0 or c.threshold <= 0:
            raise ValueError('Synaptic efficacy, learning rate and threshold must be positive')
        if not 0 < c.minimum_inhibition <= c.initial_inhibition <= c.maximum_inhibition:
            raise ValueError('Invalid inhibition bounds')
        if not 0 <= c.target_low_fraction <= c.target_high_fraction <= 1 or not 0 <= c.homeostasis_rate < 1:
            raise ValueError('Invalid population homeostasis parameters')
        self.width = c.width
        self.site_of_cell = np.arange(c.width, dtype=np.int32) // c.site_width
        self.site_count = int(self.site_of_cell[-1]) + 1
        self.site_sizes = np.bincount(self.site_of_cell, minlength=self.site_count).astype(np.float32)
        self.target_low = max(1, int(math.ceil(c.width * c.target_low_fraction)))
        self.target_high = max(self.target_low, int(math.ceil(c.width * c.target_high_fraction)))
        self.excitatory = _LazyRows(c.width, c.width)
        self.disinhibitory = _LazyRows(c.width, self.site_count)
        self.recurrent_enabled = True
        self.disinhibition_enabled = True
        self.inhibition_enabled = True
        self.external_enabled = True
        self.recalled_enabled = True
        self.advance_count = 0
        self.transition_count = 0
        self.maintenance_ticks = 0
        self.reset_activity()

    def _vector(self, value, name, upper=None):
        array = np.asarray(value, dtype=np.float32)
        if array.shape != (self.width,) or not np.isfinite(array).all() or (array < 0).any():
            raise ValueError(name + ' must be a finite nonnegative vector of width ' + str(self.width))
        if upper is not None and (array > upper).any():
            raise ValueError(name + ' exceeds the allowed activation range')
        return array

    def activity_from_indices(self, indices):
        ids = np.asarray(indices, dtype=np.int64)
        if ids.ndim != 1 or (ids < 0).any() or (ids >= self.width).any():
            raise ValueError('Invalid cell indices')
        result = np.zeros(self.width, dtype=bool)
        result[ids] = True
        return result

    def reset_activity(self):
        self.activity = np.zeros(self.width, dtype=bool)
        self.membrane = np.zeros(self.width, dtype=np.float32)
        self.adaptation = np.zeros(self.width, dtype=np.float32)
        self.inhibition_strength = self.config.initial_inhibition
        self.last_diagnostics = {}

    def set_activity(self, activity, membrane=None):
        """Explicit state intervention; no synapse or experience is modified."""
        self.activity = self._vector(activity, 'activity', 1.) > 0
        self.membrane = (np.zeros(self.width, dtype=np.float32) if membrane is None else
                         self._vector(membrane, 'membrane').copy())
        self.adaptation[:] = 0.

    def _maintenance(self):
        self.maintenance_ticks += 1
        c = self.config
        if self.maintenance_ticks % c.decay_interval == 0:
            self.excitatory.decay(c.decay_factor)
            self.disinhibitory.decay(c.decay_factor)
        self.excitatory.prune_decayed_rows(c.prune_rows_per_tick, c.disappearance_threshold)
        self.disinhibitory.prune_decayed_rows(c.prune_rows_per_tick, c.disappearance_threshold)

    def observe_transition(self, actual_before, actual_after, gain=1.):
        """Learn actual temporal coactivity regardless of task reward.

        gain is observed-transition confidence/learning strength, not a route or
        success label. Reward-dependent motivation remains a separate system.
        """
        before = self._vector(actual_before, 'actual_before', 1.)
        after = self._vector(actual_after, 'actual_after', 1.)
        gain = float(gain)
        if not np.isfinite(gain) or gain < 0:
            raise ValueError('Transition gain must be finite and nonnegative')
        pre, post = np.flatnonzero(before), np.flatnonzero(after)
        sites = np.unique(self.site_of_cell[post])
        # One vote per active target site, matching the original 去抑 learning.
        site_amplitudes = np.zeros(self.site_count, dtype=np.float32)
        np.maximum.at(site_amplitudes, self.site_of_cell[post], after[post])
        rate = self.config.hebb_rate * gain
        self.excitatory.learn(pre, before[pre], post, after[post], rate, self.config.synapse_ceiling)
        self.disinhibitory.learn(pre, before[pre], sites, site_amplitudes[sites], rate, self.config.synapse_ceiling)
        self.transition_count += 1
        self._maintenance()
        return dict(excitatory_edges=self.excitatory.edge_count,
                    disinhibitory_edges=self.disinhibitory.edge_count,
                    observed_pre_active=len(pre), observed_post_active=len(post))

    def advance(self, external_current, recalled_current=None, learning=False):
        c = self.config
        external = self._vector(external_current, 'external_current').copy()
        recalled = (np.zeros(self.width, dtype=np.float32) if recalled_current is None else
                    self._vector(recalled_current, 'recalled_current').copy())
        if not self.external_enabled:
            external[:] = 0.
        if not self.recalled_enabled:
            recalled[:] = 0.
        previous = self.activity.astype(np.float32)
        recurrent, used_excitatory = self.excitatory.propagate(previous)
        disinhibition, used_disinhibitory = self.disinhibitory.propagate(previous)
        if not self.recurrent_enabled:
            recurrent[:] = 0.
        if not self.disinhibition_enabled:
            disinhibition[:] = 0.
        local_density = np.bincount(self.site_of_cell, weights=previous,
                                   minlength=self.site_count).astype(np.float32) / self.site_sizes
        strength_before = self.inhibition_strength
        inhibitory_field = strength_before * (c.resting_inhibition + c.local_feedback_gain * local_density)
        if not self.inhibition_enabled:
            inhibitory_field[:] = 0.
        disinhibitory_field = c.disinhibition_gain * disinhibition
        net_gate = np.maximum(0., inhibitory_field - disinhibitory_field)
        self.adaptation = c.adaptation_decay * self.adaptation + (1 - c.adaptation_decay) * previous
        retained = c.membrane_decay * self.membrane
        external_drive = c.external_gain * external
        replay_drive = c.recalled_gain * recalled
        recurrent_drive = c.recurrent_gain * recurrent
        self.membrane = retained + external_drive + replay_drive + recurrent_drive
        firing_threshold = c.threshold + net_gate[self.site_of_cell] + c.adaptation_gain * self.adaptation
        self.activity = self.membrane >= firing_threshold
        count = int(self.activity.sum())
        attempted = bool(np.any(external_drive) or np.any(replay_drive) or np.any(recurrent_drive))
        if count > self.target_high:
            excess = min(4., count / self.target_high - 1.)
            self.inhibition_strength = min(c.maximum_inhibition, strength_before * (1 + c.homeostasis_rate * excess))
        elif count < self.target_low and attempted:
            self.inhibition_strength = max(c.minimum_inhibition, strength_before * (1 - c.homeostasis_rate))
        self.advance_count += 1
        if learning:
            self._maintenance()
        diagnostics = dict(external_current=external_drive, recalled_current=replay_drive,
            recurrent_current=recurrent_drive, retained_membrane_current=retained,
            inhibitory_site_current=inhibitory_field, disinhibitory_site_current=disinhibitory_field,
            net_inhibitory_site_current=net_gate, local_site_density=local_density,
            firing_threshold=firing_threshold, membrane=self.membrane.copy(), adaptation=self.adaptation.copy(),
            active_count=count, presynaptic_activity_sum=float(previous.sum()),
            inhibition_strength_before=strength_before, inhibition_strength_after=self.inhibition_strength,
            excitatory_edges=self.excitatory.edge_count, disinhibitory_edges=self.disinhibitory.edge_count,
            used_excitatory_edges=used_excitatory, used_disinhibitory_edges=used_disinhibitory,
            recurrent_current_bound=c.recurrent_gain * c.synapse_ceiling,
            allocated_synapse_bytes=self.excitatory.allocated_bytes + self.disinhibitory.allocated_bytes,
            transition_count=self.transition_count, advance_count=self.advance_count,
            removed_by_uniform_decay=self.excitatory.removed_by_decay + self.disinhibitory.removed_by_decay)
        self.last_diagnostics = diagnostics
        return self.activity.copy(), {key: value.copy() if isinstance(value, np.ndarray) else value
                                      for key, value in diagnostics.items()}

    def snapshot(self):
        arrays = dict(activity=self.activity.copy(), membrane=self.membrane.copy(),
                      adaptation=self.adaptation.copy())
        metadata = dict(snapshot_version=SNAPSHOT_VERSION, config=asdict(self.config),
            inhibition_strength=self.inhibition_strength, advance_count=self.advance_count,
            transition_count=self.transition_count, maintenance_ticks=self.maintenance_ticks,
            flags={key: getattr(self, key) for key in ('recurrent_enabled', 'disinhibition_enabled',
                'inhibition_enabled', 'external_enabled', 'recalled_enabled')}, diagnostics={})
        for prefix, table in (('excitatory', self.excitatory), ('disinhibitory', self.disinhibitory)):
            exported, state = table.export(prefix)
            arrays.update(exported)
            metadata[prefix] = state
        for key, value in self.last_diagnostics.items():
            if isinstance(value, np.ndarray):
                arrays['diagnostic_' + key] = value.copy()
            else:
                metadata['diagnostics'][key] = value
        return {'metadata': metadata, 'arrays': arrays}

    @classmethod
    def from_snapshot(cls, snapshot):
        metadata, arrays = snapshot['metadata'], snapshot['arrays']
        if metadata.get('snapshot_version') != SNAPSHOT_VERSION:
            raise ValueError('Unsupported recurrent-core snapshot')
        obj = cls(**metadata['config'])
        for key in ('activity', 'membrane', 'adaptation'):
            value = np.asarray(arrays[key])
            if value.shape != (obj.width,) or not np.isfinite(value).all():
                raise ValueError('Invalid core state: ' + key)
            setattr(obj, key, value.copy())
        for key in ('inhibition_strength', 'advance_count', 'transition_count', 'maintenance_ticks'):
            setattr(obj, key, metadata[key])
        for key, value in metadata['flags'].items():
            setattr(obj, key, bool(value))
        obj.excitatory.restore('excitatory', arrays, metadata['excitatory'])
        obj.disinhibitory.restore('disinhibitory', arrays, metadata['disinhibitory'])
        obj.last_diagnostics = dict(metadata['diagnostics'])
        for key, value in arrays.items():
            if key.startswith('diagnostic_'):
                obj.last_diagnostics[key[len('diagnostic_'):]] = np.asarray(value).copy()
        return obj


RecurrentCore = SparseHebbianPFC
