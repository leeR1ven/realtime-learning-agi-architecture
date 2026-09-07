"""A transparent, continuous four-actuator sound body, with no language model.

This is a *signal-level approximation*, not an anatomical vocal tract. A
phase-continuous harmonic source has a smoothly changing spectral envelope.
Controls are [airflow, tension, aperture, tongue], each in [0, 1]. No text,
phoneme, word, target frequency label, or recorded voice enters this module.

The fixed body constants are engineering choices. Their purpose is to give
motor actions repeatable audible consequences that a brain can experience.
The state belongs to the physical body; it adds no neural memory or clock.
"""
from __future__ import annotations

from pathlib import Path
import wave

import numpy as np

from checkpoint import load_tree, save_tree


SAMPLE_RATE = 8000
FRAME_SAMPLES = 800
CONTROL_COUNT = 4
CONTROL_NAMES = ("airflow", "tension", "aperture", "tongue")
BODY_KIND = "continuous_harmonic_vocal_body_v1"
SCHEMA_VERSION = 1


def _controls(value):
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (CONTROL_COUNT,) or not np.isfinite(result).all():
        raise ValueError("vocal controls must be a finite vector of length four")
    if np.any((result < 0.) | (result > 1.)):
        raise ValueError("vocal controls must be within 0..1")
    return result.copy()


def _pcm(value, name):
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (FRAME_SAMPLES,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain exactly 800 finite 8 kHz samples")
    if np.any(np.abs(result) > 1.):
        raise ValueError(f"{name} must be within -1..1")
    return result.copy()


class VocalBody:
    """Motor action -> actual float64 PCM, one 100 ms frame at a time.

    ``render_frame`` advances the body once. The returned waveform should be
    mixed into the *next* sensory observation, like the other effects of the
    action just executed. ``snapshot`` can be embedded inside the existing
    complete life checkpoint; it contains no learned or indexed memories.
    """

    # A fixed finite harmonic bank gives a strict, parameter-independent bound.
    _harmonics = np.arange(1, 46, dtype=np.float64)[:, None]
    _source_weights = _harmonics ** -1.25
    _normalizer = float(_source_weights.sum())
    _slew_seconds = .012

    def __init__(self):
        self.phase = 0.  # Fundamental phase, radians in [0, 2*pi).
        self.actuators = np.zeros(CONTROL_COUNT, dtype=np.float64)
        self.samples_rendered = 0

    @staticmethod
    def parameters(controls):
        """Inspect the body's fixed control-to-acoustic parameter mapping."""
        c = _controls(controls)
        return {
            "airflow": float(c[0]),
            "fundamental_hz": float(80. + 260. * c[1]),
            "first_resonance_hz": float(250. + 650. * c[2]),
            "second_resonance_hz": float(900. + 1500. * c[3]),
            "first_bandwidth_hz": 130.,
            "second_bandwidth_hz": 210.,
        }

    def render_frame(self, controls):
        return self.render_samples(controls, FRAME_SAMPLES)

    def render_samples(self, controls, sample_count):
        """General block renderer; normal brain integration uses 800 samples.

        Public for checking arbitrary block partition continuity. A block is
        capped at one second to keep memory bounded. Invalid input never
        advances phase, actuator position, or elapsed physical samples.
        """
        command = _controls(controls)
        if type(sample_count) is not int or not 1 <= sample_count <= SAMPLE_RATE:
            raise ValueError("sample_count must be an integer in 1..8000")
        positions = np.arange(1, sample_count + 1, dtype=np.float64)
        step = 1. / (SAMPLE_RATE * self._slew_seconds)
        decay = np.exp(-positions * step)
        actual = command[None, :] + (self.actuators - command)[None, :] * decay[:, None]
        frequency = 80. + 260. * actual[:, 1]
        # Integrate the exact exponential actuator trajectory analytically.
        # This avoids accumulating thousands of float additions per frame.
        decay_sum = -np.expm1(-positions * step) / np.expm1(step)
        turns = ((80. + 260. * command[1]) * positions
                 + 260. * (self.actuators[1] - command[1]) * decay_sum) / SAMPLE_RATE
        phases = (self.phase + 2. * np.pi * turns) % (2. * np.pi)
        harmonic_hz = self._harmonics * frequency[None, :]
        f1 = 250. + 650. * actual[:, 2]
        f2 = 900. + 1500. * actual[:, 3]
        resonance1 = np.exp(-.5 * ((harmonic_hz - f1[None, :]) / 130.) ** 2)
        resonance2 = np.exp(-.5 * ((harmonic_hz - f2[None, :]) / 210.) ** 2)
        # A cosine taper removes harmonics before Nyquist. Modulated signals
        # still have sidebands: this is not a claim of an ideal anti-alias filter.
        rolloff = np.clip((harmonic_hz - 3200.) / 400., 0., 1.)
        antialias = .5 * (1. + np.cos(np.pi * rolloff))
        base = self._source_weights * antialias
        first = base * resonance1
        second = base * resonance2
        # Three fixed source/formant groups share a unit amplitude budget.
        # Normalization is a smooth function of actuator geometry and source
        # frequency, not of measured frame loudness or audio/text content.
        # Without resonance gain, the upper band would be inaudible to the
        # existing coarse auditory receptors even at maximum airflow.
        weights = (.12 * base / self._normalizer
                   + .60 * first / np.maximum(first.sum(axis=0), 1e-12)
                   + .28 * second / np.maximum(second.sum(axis=0), 1e-12))
        pressure = np.sum(weights * np.sin(self._harmonics * phases[None, :]), axis=0)
        result = .95 * actual[:, 0] * pressure
        if not np.isfinite(result).all() or np.max(np.abs(result)) > 1.:
            raise RuntimeError("vocal body produced invalid PCM; body state was not advanced")
        self.phase = float(phases[-1] % (2. * np.pi))
        self.actuators = actual[-1].copy()
        self.samples_rendered += sample_count
        return result

    def snapshot(self):
        return {
            "kind": BODY_KIND, "schema_version": SCHEMA_VERSION,
            "sample_rate": SAMPLE_RATE, "phase": self.phase,
            "actuators": self.actuators.copy(),
            "samples_rendered": self.samples_rendered,
        }

    @classmethod
    def from_snapshot(cls, state):
        fields = {"kind", "schema_version", "sample_rate", "phase", "actuators", "samples_rendered"}
        if not isinstance(state, dict) or set(state) != fields:
            raise ValueError("invalid vocal body state fields")
        if state["kind"] != BODY_KIND or type(state["schema_version"]) is not int or state["schema_version"] != SCHEMA_VERSION:
            raise ValueError("unsupported vocal body state")
        if type(state["sample_rate"]) is not int or state["sample_rate"] != SAMPLE_RATE:
            raise ValueError("vocal body sample rate mismatch")
        phase = state["phase"]
        if type(phase) not in (int, float) or not np.isfinite(phase) or not 0. <= phase < 2. * np.pi:
            raise ValueError("invalid vocal body phase")
        elapsed = state["samples_rendered"]
        if type(elapsed) is not int or elapsed < 0:
            raise ValueError("invalid vocal body elapsed samples")
        actuators = _controls(state["actuators"])
        result = cls()
        result.phase, result.actuators, result.samples_rendered = float(phase), actuators, elapsed
        return result

    def save(self, path):
        save_tree(path, self.snapshot())

    @classmethod
    def load(cls, path):
        return cls.from_snapshot(load_tree(path))


def mix_at_ear(environment_pcm, self_pcm, *, environment_gain=.65, self_gain=.35):
    """A fixed linear acoustic mixture, not a tagged self/other neural channel.

    Nonnegative gains with a sum <= 1 guarantee valid PCM without clipping or
    automatic normalization. Setting self_gain=0 is a physical deaf-to-self
    ablation; it does not touch the model's neural state.
    """
    external = _pcm(environment_pcm, "environment PCM")
    own = _pcm(self_pcm, "self PCM")
    gains = np.asarray([environment_gain, self_gain], dtype=np.float64)
    if not np.isfinite(gains).all() or np.any(gains < 0.) or gains.sum() > 1.:
        raise ValueError("ear gains must be finite, nonnegative, and sum to at most one")
    return float(gains[0]) * external + float(gains[1]) * own


def write_pcm16(path, waveform):
    """Export actual generated sound for inspection; never reads a recording."""
    pcm = np.asarray(waveform, dtype=np.float64)
    if pcm.ndim != 1 or not len(pcm) or not np.isfinite(pcm).all() or np.any(np.abs(pcm) > 1.):
        raise ValueError("WAV output must be a nonempty finite -1..1 vector")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(target), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(SAMPLE_RATE)
        stream.writeframes(np.rint(pcm * 32767.).astype('<i2').tobytes())
