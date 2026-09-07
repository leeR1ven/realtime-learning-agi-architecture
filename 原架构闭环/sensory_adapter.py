"""Physical sensors and four muscles adapted to the ORIGINAL neural classes.

Only the classes/functions are extracted, unchanged, from the three original
files. Their local circular source tables, fixed uniform birth weights,
threshold propagation, paired receptors and reverse Hebbian motor learning are
used directly. Global 24x16 image instances and demonstration code are not run.

There are no target-frequency detectors, goal/rule groups, color labels, reward
tables or action classes here. The caller must call learn_executed only AFTER
the supplied muscles really executed. encode and decode_motor never learn.
"""
from __future__ import annotations

import ast
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SAMPLE_RATE = 8000
FRAME_SAMPLES = 800
RAY_COUNT = 31
MUSCLE_COUNT = 4
SCHEMA_VERSION = 1
ADAPTER_KIND = "original_local_threshold_sensory_motor_v1"


class _LocalNumpy:
    """Original code calls np.random.uniform; isolate its birth RNG only."""
    def __init__(self, seed):
        self.random = np.random.RandomState(seed)

    def __getattr__(self, name):
        return getattr(np, name)


@lru_cache(maxsize=3)
def _original_definition(filename):
    path = ROOT / filename
    source = path.read_bytes()
    tree = ast.parse(source.decode("utf-8-sig"), filename=str(path))
    wanted = {"特征神经元池", "神经网络"}
    if filename == "运动输出区.py":
        wanted.update(("反向赫布网络", "发力转信号", "解码发力"))
    nodes = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))
             and n.name in wanted]
    if {n.name for n in nodes} != wanted:
        raise RuntimeError("original neural interfaces are missing: " + filename)
    module = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    return compile(module, str(path), "exec"), hashlib.sha256(source).hexdigest()


def _finite_array(value, shape, name, *, low=None, high=None):
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite with shape {shape}")
    if low is not None and np.any(array < low):
        raise ValueError(f"{name} must be >= {low}")
    if high is not None and np.any(array > high):
        raise ValueError(f"{name} must be <= {high}")
    return array.copy()


def _json_bytes(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False,
                      sort_keys=True, separators=(",", ":")).encode("utf-8")


class SensoryAdapter:
    def __init__(self, *, seed=2026, audio_bands=128, levels=10, layers=3,
                 ray_count=RAY_COUNT, connection_radius=10, weight_range=.1, threshold=.6,
                 motor_once=True, motor_background=.05, distance_scale=16.,
                 visual_threshold=None, audio_threshold=None, motor_threshold=None,
                 audio_levels=None, motor_levels=None, motor_reverse_window=None,
                 contrast_to_silence=False):
        integer_options = dict(seed=seed, audio_bands=audio_bands, levels=levels,
                               layers=layers, connection_radius=connection_radius)
        if any(type(v) is not int for v in integer_options.values()):
            raise ValueError("seed/bands/levels/layers/radius/ray_count must be integers")
        if not (0 <= seed < 2**32 and 8 <= audio_bands <= 400 and 2 <= levels <= 64
                and 2 <= layers <= 8 and 1 <= connection_radius <= 40
                and 8 <= ray_count <= 512):
            raise ValueError("invalid neural layout")
        if not np.isfinite([weight_range, threshold, motor_background, distance_scale]).all() or not (
                weight_range > 0 and threshold > 0 and motor_background >= 0 and distance_scale > 0):
            raise ValueError("invalid birth weight/threshold/background")
        if type(motor_once) is not bool:
            raise ValueError("motor_once must be bool")
        if type(contrast_to_silence) is not bool:
            raise ValueError("contrast_to_silence must be bool")
        self.config = dict(**integer_options, weight_range=float(weight_range),
                           threshold=float(threshold), motor_once=motor_once,
                           motor_background=float(motor_background), distance_scale=float(distance_scale))
        self.levels = levels
        self.ray_count = int(ray_count)
        if self.ray_count != RAY_COUNT:
            self.config["ray_count"] = self.ray_count
        self.contrast_to_silence = contrast_to_silence
        if contrast_to_silence:
            self.config["contrast_to_silence"] = True
        self.audio_levels = levels if audio_levels is None else audio_levels
        self.motor_levels = levels if motor_levels is None else motor_levels
        for name, value in (("audio_levels", audio_levels), ("motor_levels", motor_levels)):
            if value is not None:
                if type(value) is not int or not 2 <= value <= 64:
                    raise ValueError(name + " must be an integer from 2 to 64")
                self.config[name] = value
        for name, value in (("visual_threshold", visual_threshold), ("audio_threshold", audio_threshold),
                            ("motor_threshold", motor_threshold)):
            if value is not None:
                if not np.isfinite(value) or value <= 0:
                    raise ValueError(name + " must be positive and finite")
                self.config[name] = float(value)
        if motor_reverse_window is not None:
            if type(motor_reverse_window) is not int or motor_reverse_window < 0:
                raise ValueError("motor_reverse_window must be a nonnegative integer or None")
            self.config["motor_reverse_window"] = motor_reverse_window
        self.audio_bands = audio_bands
        proxy = _LocalNumpy(seed)
        self.original_source_hashes = {}
        namespaces = {}
        for name in ("视觉前处理.py", "听觉前处理.py", "运动输出区.py"):
            compiled, sha = _original_definition(name)
            namespace = {"np": proxy}
            exec(compiled, namespace)
            namespaces[name] = namespace
            self.original_source_hashes[name] = sha
        self.pools, self.networks = {}, {}
        self.scalar_widths = dict(rgb=self.ray_count * 3, depth=self.ray_count, body=8,
                                  audio=audio_bands, motor=MUSCLE_COUNT)
        self.scalar_levels = {name: self.audio_levels if name == "audio" else
                              self.motor_levels if name == "motor" else levels
                              for name in self.scalar_widths}
        for name, width in self.scalar_widths.items():
            namespace = namespaces["运动输出区.py" if name == "motor" else
                                   "听觉前处理.py" if name == "audio" else "视觉前处理.py"]
            pool = namespace["特征神经元池"](width * self.scalar_levels[name])
            local_threshold = (motor_threshold if name == "motor" else
                               audio_threshold if name == "audio" else visual_threshold)
            network = namespace["神经网络"](pool, 层数=layers,
                连接半径=connection_radius, 权重范围=weight_range,
                阈值初值=threshold if local_threshold is None else local_threshold)
            self.pools[name], self.networks[name] = pool, network
        self._motor_namespace = namespaces["运动输出区.py"]
        self._motor_namespace.update(肌肉数=MUSCLE_COUNT, 每档对数=self.motor_levels,
                                     池=self.pools["motor"])
        self.motor_reverse_pool = self._motor_namespace["特征神经元池"](MUSCLE_COUNT * self.motor_levels)
        self.motor_reverse = self._motor_namespace["反向赫布网络"](
            self.motor_reverse_pool, 窗口距离=motor_reverse_window, 本底范围=motor_background, 一次学会=motor_once)
        self.widths = dict(visual=sum(self.pools[name].总数 for name in ("rgb", "depth", "body")),
                           audio=self.pools["audio"].总数, motor=self.pools["motor"].总数)
        self.visual_segments = {}
        start = 0
        for name in ("rgb", "depth", "body"):
            end = start + self.pools[name].总数
            self.visual_segments[name] = (start, end)
            start = end
        self.audio_band_edges_hz = np.linspace(0., SAMPLE_RATE / 2, audio_bands + 1)
        self._fft_band = np.minimum((np.fft.rfftfreq(FRAME_SAMPLES, 1 / SAMPLE_RATE)
                                    * audio_bands / (SAMPLE_RATE / 2)).astype(np.int32),
                                   audio_bands - 1)
        self._window = np.hanning(FRAME_SAMPLES)
        self.motor_learning_steps = 0
        self.audio_silence_baseline = None
        if self.contrast_to_silence:
            # Explicitly authorized NEW fixed contrast layer. Original auditory
            # network and source cells are unchanged. A cell is active when its
            # original response differs from its own fixed silent response.
            signal = self.thermometer(self.spectrum(np.zeros(FRAME_SAMPLES)), self.audio_levels)
            self.audio_silence_baseline = self.networks["audio"].前向传播(signal).astype(bool, copy=True)
            self.reset_activity()
        self.birth_hash = self._birth_hash()

    def _fixed_arrays(self):
        result = {}
        for name, network in self.networks.items():
            for layer in range(network.层数 - 1):
                for english, chinese in (("sources", "来源"), ("weights", "权重"), ("thresholds", "阈值")):
                    result[f"fixed_{name}_{english}_{layer}"] = getattr(network, chinese)[layer]
        result["fixed_motor_background"] = self.motor_reverse.本底
        result["fixed_audio_band_edges_hz"] = self.audio_band_edges_hz
        result["fixed_fft_band"] = self._fft_band
        result["fixed_fft_window"] = self._window
        if self.audio_silence_baseline is not None:
            result["fixed_audio_silence_baseline"] = self.audio_silence_baseline
        return result

    def _birth_hash(self):
        value = hashlib.sha256(_json_bytes(dict(kind=ADAPTER_KIND, config=self.config,
                                               originals=self.original_source_hashes)))
        for name, array in sorted(self._fixed_arrays().items()):
            value.update(name.encode())
            value.update(str(array.dtype).encode())
            value.update(np.ascontiguousarray(array).tobytes())
        return value.hexdigest()

    def thermometer(self, values, levels=None):
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 1 or not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
            raise ValueError("thermometer values must be a finite 0..1 vector")
        count = self.levels if levels is None else levels
        ticks = np.rint(values * count).astype(np.int32)
        positive = np.arange(count)[None, :] < ticks[:, None]
        return np.stack((positive, ~positive), axis=-1).ravel().astype(np.float64)

    def spectrum(self, waveform):
        wave = np.asarray(waveform, dtype=np.float64)
        # Empty means the ordinary absence of a packet of sound, encoded silence.
        if wave.shape == (0,):
            wave = np.zeros(FRAME_SAMPLES, dtype=np.float64)
        wave = _finite_array(wave, (FRAME_SAMPLES,), "8 kHz PCM", low=-1., high=1.)
        magnitude = 2 * np.abs(np.fft.rfft(wave * self._window)) / self._window.sum()
        magnitude[[0, -1]] *= .5
        energy = np.bincount(self._fft_band, weights=magnitude**2, minlength=self.audio_bands)
        amplitude = np.sqrt(energy)
        # Same gain/compression for EVERY band. No per-frequency learned labels.
        return np.clip(np.log1p(9 * amplitude) / np.log(10), 0., 1.)

    def _scalars(self, packet, muscles):
        if not isinstance(packet, dict) or not isinstance(packet.get("observation"), dict):
            raise ValueError("packet must contain a sensory observation dictionary")
        observation = packet["observation"]
        rgb = _finite_array(observation["ray_colors"], (self.ray_count, 3), "ray RGB", low=0, high=1)
        distances = _finite_array(observation["ray_distances"], (self.ray_count,), "ray distances", low=0)
        velocity = _finite_array(observation["velocity"], (2,), "body velocity")
        angular = _finite_array(observation["angular_velocity"], (), "angular velocity")
        pain = _finite_array(observation.get("pain", 0.), (), "pain", low=0, high=1)
        touch = np.asarray(observation.get("touch", np.zeros(4)), dtype=np.float64)
        if touch.shape == ():
            touch = np.repeat(touch, 4)
        touch = _finite_array(touch, (4,), "four touch sensors", low=0, high=1)
        actual = _finite_array(muscles, (4,), "last executed muscles", low=0, high=1)
        return dict(rgb=rgb.ravel(), depth=np.clip(distances / self.config["distance_scale"], 0., 1.),
            body=np.r_[np.clip((velocity + 3.) / 6., 0., 1.),
                       np.clip((angular + 4.) / 8., 0., 1.), pain, touch],
            audio=self.spectrum(packet.get("waveform", ())), motor=actual)

    def encode(self, packet, last_executed_muscles):
        """Encode only whitelisted real sensors; no learning, goal or map inputs."""
        scalars = self._scalars(packet, last_executed_muscles)
        signals = {name: self.thermometer(value, self.scalar_levels[name]) for name, value in scalars.items()}
        signals["motor"] = self._motor_namespace["发力转信号"](scalars["motor"])
        outputs = {name: network.前向传播(signals[name]).astype(bool, copy=True)
                   for name, network in self.networks.items()}
        raw_audio = outputs["audio"].copy()
        if self.contrast_to_silence:
            outputs["audio"] = np.logical_xor(raw_audio, self.audio_silence_baseline)
        inputs = {name: array.copy() for name, array in signals.items()}
        inputs["visual"] = np.concatenate([signals[name] for name in ("rgb", "depth", "body")])
        result = dict(visual=np.concatenate([outputs[name] for name in ("rgb", "depth", "body")]),
                    audio=outputs["audio"], motor=outputs["motor"], input_arrays=inputs,
                    scalars={name: value.copy() for name, value in scalars.items()},
                    birth_hash=self.birth_hash,
                    diagnostics={name: dict(output_active=int(outputs[name].sum()),
                        layer_active=[int(a.sum()) for a in network.普通层激活])
                        for name, network in self.networks.items()})
        result["raw_outputs"] = {"audio": raw_audio}
        result["diagnostics"]["audio"].update(contrast_to_silence=self.contrast_to_silence,
            original_output_active=int(raw_audio.sum()), original_output_ids=np.flatnonzero(raw_audio).tolist())
        if self.contrast_to_silence:
            positive = raw_audio & ~self.audio_silence_baseline
            negative = self.audio_silence_baseline & ~raw_audio
            result["diagnostics"]["audio"].update(
                silence_baseline_active=int(self.audio_silence_baseline.sum()),
                silence_baseline_sha256=hashlib.sha256(self.audio_silence_baseline.tobytes()).hexdigest(),
                positive_deviation_ids=np.flatnonzero(positive).tolist(),
                negative_deviation_ids=np.flatnonzero(negative).tolist())
        return result

    def learn_executed(self, actual_muscles):
        """Call after actual physics execution, never for an imagined command."""
        muscles = _finite_array(actual_muscles, (4,), "executed muscles", low=0, high=1)
        signal = self._motor_namespace["发力转信号"](muscles)
        code = self.networks["motor"].前向传播(signal).astype(bool, copy=True)
        self.motor_reverse.学习一帧(self.pools["motor"].激活, code)
        self.motor_learning_steps += 1
        return code

    def learn_motor_from_encoded(self, encoded):
        """Same original update, reusing encode of the last actual execution."""
        if encoded.get("birth_hash") != self.birth_hash:
            raise ValueError("encoded input belongs to another birth")
        raw = _finite_array(encoded["input_arrays"]["motor"], (self.widths["motor"],), "motor receptors", low=0, high=1)
        code = _finite_array(encoded["motor"], (self.widths["motor"],), "motor code", low=0, high=1)
        if not np.isin(raw, [0, 1]).all() or not np.isin(code, [0, 1]).all() or not np.all(raw.reshape(-1, 2).sum(axis=1) == 1):
            raise ValueError("expected original paired motor receptors and binary code")
        self.motor_reverse.学习一帧(raw.astype(bool), code.astype(bool))
        self.motor_learning_steps += 1

    def decode_motor(self, motor_code):
        """Original reverse-Hebb votes + paired competition, including its rest noise."""
        code = _finite_array(motor_code, (self.widths["motor"],), "motor code", low=0, high=1)
        if not np.isin(code, [0, 1]).all():
            raise ValueError("motor code must be binary")
        activity = self.motor_reverse.前向(code.astype(bool)).copy()
        return (self._motor_namespace["解码发力"](activity) / self.motor_levels).astype(np.float64)

    def reset_activity(self):
        for pool in [*self.pools.values(), self.motor_reverse_pool]:
            pool.信号[:] = 0
            pool.激活[:] = False
        for network in self.networks.values():
            for activity in network.普通层激活:
                activity[:] = False

    def snapshot(self):
        arrays = {name: array.copy() for name, array in self._fixed_arrays().items()}
        for name, pool in self.pools.items():
            arrays[f"state_{name}_signal"] = pool.信号.copy()
            arrays[f"state_{name}_receptors"] = pool.激活.copy()
            for layer, activity in enumerate(self.networks[name].普通层激活):
                arrays[f"state_{name}_layer_{layer}"] = activity.copy()
        arrays["plastic_motor_reverse"] = self.motor_reverse.矩阵.copy()
        arrays["state_reverse_signal"] = self.motor_reverse_pool.信号.copy()
        arrays["state_reverse_receptors"] = self.motor_reverse_pool.激活.copy()
        metadata = dict(schema_version=SCHEMA_VERSION, adapter_kind=ADAPTER_KIND,
            config=self.config.copy(), birth_hash=self.birth_hash, widths=self.widths.copy(),
            original_source_hashes=self.original_source_hashes.copy(),
            motor_learning_steps=self.motor_learning_steps,
            motor_reverse_connections=self.motor_reverse.学过的条数)
        return dict(metadata=metadata, arrays=arrays)

    @classmethod
    def from_snapshot(cls, snapshot):
        metadata, arrays = snapshot["metadata"], snapshot["arrays"]
        if metadata.get("schema_version") != SCHEMA_VERSION or metadata.get("adapter_kind") != ADAPTER_KIND:
            raise ValueError("not an original sensory/motor adapter snapshot")
        config = dict(metadata["config"])
        if config.get("ray_count") == RAY_COUNT:
            config.pop("ray_count")
        obj = cls(**config)
        if metadata["birth_hash"] != obj.birth_hash or metadata["original_source_hashes"] != obj.original_source_hashes or metadata["widths"] != obj.widths:
            raise ValueError("birth/source/layout mismatch")
        expected = obj.snapshot()["arrays"]
        if arrays.keys() != expected.keys():
            raise ValueError("snapshot array keys differ")
        for name, default in expected.items():
            value = np.asarray(arrays[name])
            if value.shape != default.shape or value.dtype != default.dtype or not np.isfinite(value).all():
                raise ValueError("invalid array: " + name)
            if name.startswith("fixed_") and not np.array_equal(value, default):
                raise ValueError("fixed original birth array changed: " + name)
        for name, pool in obj.pools.items():
            pool.信号[:] = arrays[f"state_{name}_signal"]
            pool.激活[:] = arrays[f"state_{name}_receptors"]
            for layer, activity in enumerate(obj.networks[name].普通层激活):
                activity[:] = arrays[f"state_{name}_layer_{layer}"]
        obj.motor_reverse.矩阵[:] = arrays["plastic_motor_reverse"]
        obj.motor_reverse_pool.信号[:] = arrays["state_reverse_signal"]
        obj.motor_reverse_pool.激活[:] = arrays["state_reverse_receptors"]
        count = int(metadata["motor_reverse_connections"])
        steps = int(metadata["motor_learning_steps"])
        if count != np.count_nonzero(obj.motor_reverse.矩阵) or steps < 0:
            raise ValueError("invalid motor learning counters")
        obj.motor_reverse.学过的条数 = count
        obj.motor_learning_steps = steps
        return obj

    def save(self, path):
        target = Path(path).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        snapshot = self.snapshot()
        arrays = snapshot["arrays"]
        arrays["__metadata_json__"] = np.frombuffer(_json_bytes(snapshot["metadata"]), np.uint8)
        fd, temporary = tempfile.mkstemp(prefix="." + target.name, suffix=".tmp", dir=target.parent)
        try:
            with os.fdopen(fd, "wb") as stream:
                np.savez_compressed(stream, **arrays)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, target)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    @classmethod
    def load(cls, path):
        with np.load(Path(path), allow_pickle=False) as archive:
            metadata = json.loads(archive["__metadata_json__"].tobytes().decode("utf-8"))
            arrays = {name: archive[name].copy() for name in archive.files if name != "__metadata_json__"}
        return cls.from_snapshot(dict(metadata=metadata, arrays=arrays))


Adapter = SensoryAdapter
