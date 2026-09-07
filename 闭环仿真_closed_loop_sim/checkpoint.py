"""Atomic, pickle-free checkpoints and conservative averaging of aligned brains.

Array contract for merge_checkpoints:
  fixed_*             identical arrays (birth projections, feature identities).
  plastic_*           floating arrays, averaged element by element.
  state_*             runtime arrays: saved/restored, but zeroed by fusion.
  mem_event_tick      int64[T], event-local chronological ticks.
  mem_event_strength  float[T], concatenated with strength / parent count.
  mem_event_next      optional int64[T], next event index or -1 for a boundary.
  mem_event_*         other event-aligned arrays; concatenate without changes.
  mem_edge_source     int64[E], feature neuron index.
  mem_edge_time       int64[E], event index, 0 <= index < T.
  mem_edge_weight     float[E], edge value, preserved (strength scales the event).
  mem_edge_target     optional int64[E], target feature neuron index.

Alternatively mem_edges float64[E,3] stores [source, time, weight]. Do not mix
packed and separate edge representations. Each parent's event IDs occupy a
distinct range after fusion; no memories or connections are deduplicated.
Event ticks are local to their original stream, and are intentionally retained.
Unknown array prefixes are rejected by the merger instead of silently combining
runtime state. model_config and state_defaults metadata must match and survive
fusion; arbitrary parent runtime metadata is not inherited. Metadata must contain birth_hash and
feature_schema; schema_version defaults to SCHEMA_VERSION when saving.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable

import numpy as np


SCHEMA_VERSION = 1
MEMORY_PROTOCOL = "event_edges_v1"
_META_KEY = "__metadata_json__"
_MAX_METADATA_BYTES = 8 * 1024 * 1024


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, allow_nan=False,
                      sort_keys=True, separators=(",", ":")).encode("utf-8")


def _metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(metadata, dict):
        raise TypeError("metadata must be a JSON-compatible dictionary")
    result = dict(metadata)
    result.setdefault("schema_version", SCHEMA_VERSION)
    result.setdefault("memory_protocol", MEMORY_PROTOCOL)
    if type(result["schema_version"]) is not int or result["schema_version"] != SCHEMA_VERSION:
        raise ValueError("unsupported checkpoint schema_version")
    if result["memory_protocol"] != MEMORY_PROTOCOL:
        raise ValueError("unsupported memory_protocol")
    if not isinstance(result.get("birth_hash"), str) or not result["birth_hash"]:
        raise ValueError("metadata.birth_hash must identify the shared birth configuration")
    if "feature_schema" not in result or result["feature_schema"] is None:
        raise ValueError("metadata.feature_schema is required")
    encoded = _json_bytes(result)
    if len(encoded) > _MAX_METADATA_BYTES:
        raise ValueError("metadata exceeds 8 MiB limit")
    # Round-tripping makes the returned representation strictly JSON-native.
    return json.loads(encoded)


def _arrays(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    if not isinstance(arrays, dict):
        raise TypeError("arrays must be a dictionary of NumPy arrays")
    result = {}
    for name, value in arrays.items():
        if not isinstance(name, str) or not name or name == _META_KEY or "/" in name or "\\" in name:
            raise ValueError(f"invalid array name: {name!r}")
        if not isinstance(value, np.ndarray):
            raise TypeError(f"{name}: expected a NumPy array")
        if value.dtype.kind not in "biufc":
            raise ValueError(f"{name}: only numeric or boolean arrays are permitted")
        if value.dtype.kind in "fc" and not np.isfinite(value).all():
            raise ValueError(f"{name}: non-finite values cannot be checkpointed")
        result[name] = value
    return result


def _integer_array(name: str, value: np.ndarray, *, count: int | None = None) -> None:
    if value.dtype != np.dtype("int64") or value.ndim != 1:
        raise ValueError(f"{name}: expected a one-dimensional int64 array")
    if count is not None and len(value) != count:
        raise ValueError(f"{name}: row count mismatch")


def _validate_memory(arrays: dict[str, np.ndarray]) -> int:
    names = {key for key in arrays if key.startswith("mem_")}
    if not names:
        return 0
    required = {"mem_event_tick", "mem_event_strength"}
    if not required <= names:
        raise ValueError("memory requires mem_event_tick and mem_event_strength")
    ticks = arrays["mem_event_tick"]
    _integer_array("mem_event_tick", ticks)
    count = len(ticks)
    strength = arrays["mem_event_strength"]
    if strength.ndim != 1 or len(strength) != count or strength.dtype.kind != "f" or (strength < 0).any():
        raise ValueError("mem_event_strength: expected nonnegative floating values, one per event")
    for name in names:
        if name.startswith("mem_event_"):
            if arrays[name].ndim == 0 or len(arrays[name]) != count:
                raise ValueError(f"{name}: expected one row per event")
    if "mem_event_next" in names:
        nxt = arrays["mem_event_next"]
        _integer_array("mem_event_next", nxt, count=count)
        if ((nxt < -1) | (nxt >= count)).any():
            raise ValueError("mem_event_next: event index out of range")
    typed = {"mem_edge_source", "mem_edge_time", "mem_edge_weight"}
    packed = "mem_edges" in names
    has_typed = bool(names & (typed | {"mem_edge_target"}))
    if packed and has_typed:
        raise ValueError("do not mix packed and separate memory edge representations")
    if has_typed:
        if not typed <= names:
            raise ValueError("memory edges require source, time, and weight arrays")
        size = len(arrays["mem_edge_time"])
        for name in ("mem_edge_source", "mem_edge_time", "mem_edge_target"):
            if name in names:
                _integer_array(name, arrays[name], count=size)
                if (arrays[name] < 0).any():
                    raise ValueError(f"{name}: negative index")
        if (arrays["mem_edge_time"] >= count).any():
            raise ValueError("mem_edge_time: event index out of range")
        weight = arrays["mem_edge_weight"]
        if weight.ndim != 1 or len(weight) != size or weight.dtype.kind != "f":
            raise ValueError("mem_edge_weight: expected one floating value per edge")
    if packed:
        edges = arrays["mem_edges"]
        if edges.dtype != np.dtype("float64") or edges.ndim != 2 or edges.shape[1] != 3:
            raise ValueError("mem_edges: expected float64[E,3] [source,time,weight]")
        indices = edges[:, :2]
        if ((indices < 0) | (indices >= 2**53) | (indices != np.floor(indices))).any():
            raise ValueError("mem_edges: indices must be exact nonnegative integers below 2**53")
        if (edges[:, 1] >= count).any():
            raise ValueError("mem_edges: event index out of range")
    allowed = typed | {"mem_edge_target", "mem_edges"}
    unknown = [key for key in names if not key.startswith("mem_event_") and key not in allowed]
    if unknown:
        raise ValueError(f"unknown memory arrays: {sorted(unknown)}")
    return count


def save_checkpoint(path: str | os.PathLike, arrays: dict[str, np.ndarray],
                    metadata: dict[str, Any]) -> None:
    """Validate then atomically replace path; a failed write leaves the old file.

    Callers must provide a stable snapshot (or hold their model lock) while this
    function runs. It deliberately does not copy an entire model a second time.
    The temporary file is in the destination directory and is flushed before
    replacement. Opening the temporary file ourselves avoids NumPy's automatic
    .npz suffix. No object arrays or pickle payloads are written.
    """
    clean_arrays = _arrays(arrays)
    _validate_memory(clean_arrays)
    clean_metadata = _metadata(metadata)
    encoded = np.frombuffer(_json_bytes(clean_metadata), dtype=np.uint8)
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        descriptor, temp_path = tempfile.mkstemp(prefix=f".{destination.name}.",
                                               suffix=".tmp", dir=destination.parent)
        with os.fdopen(descriptor, "wb") as handle:
            np.savez(handle, **clean_arrays, **{_META_KEY: encoded})
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, destination)
        temp_path = None
    finally:
        if temp_path is not None:
            try:
                Path(temp_path).unlink(missing_ok=True)
            except OSError:
                pass


def load_checkpoint(path: str | os.PathLike) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Load a validated checkpoint without enabling pickle."""
    with np.load(Path(path), allow_pickle=False) as archive:
        if _META_KEY not in archive.files:
            raise ValueError("checkpoint has no JSON metadata")
        if len(archive.files) != len(set(archive.files)):
            raise ValueError("checkpoint contains duplicate array names")
        raw = archive[_META_KEY]
        if raw.dtype != np.uint8 or raw.ndim != 1 or len(raw) > _MAX_METADATA_BYTES:
            raise ValueError("invalid checkpoint metadata encoding")
        metadata = _metadata(json.loads(raw.tobytes().decode("utf-8")))
        arrays = _arrays({key: archive[key] for key in archive.files if key != _META_KEY})
    _validate_memory(arrays)
    return arrays, metadata


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def merge_checkpoints(paths: Iterable[str | os.PathLike], out_path: str | os.PathLike) -> dict[str, Any]:
    """Average compatible parents, concatenate every temporal event, and save.

    Returns the output metadata. Original parents are never overwritten. Parent
    files must stay unchanged during this call; their hashes are checked around
    loading. Input multiplicity is rejected to avoid unintentionally counting
    the same parent twice. Fixed arrays are compared exactly, not approximately.
    """
    parents = [Path(path).expanduser().resolve() for path in paths]
    destination = Path(out_path).expanduser().resolve()
    if not parents:
        raise ValueError("at least one parent checkpoint is required")
    if len(set(parents)) != len(parents):
        raise ValueError("duplicate parent checkpoint paths")
    if destination in parents:
        raise ValueError("fusion output must not overwrite a parent checkpoint")
    loaded, provenance = [], []
    for parent in parents:
        before = _file_hash(parent)
        arrays, metadata = load_checkpoint(parent)
        after = _file_hash(parent)
        if before != after:
            raise RuntimeError(f"parent checkpoint changed while loading: {parent}")
        loaded.append((arrays, metadata))
        provenance.append({"path": str(parent), "sha256": after})
    reference, reference_meta = loaded[0]
    keys = set(reference)
    for arrays, metadata in loaded:
        for key in ("schema_version", "birth_hash", "feature_schema", "memory_protocol"):
            if _json_bytes(metadata[key]) != _json_bytes(reference_meta[key]):
                raise ValueError(f"parent {key} mismatch")
        for key in ("model_config", "state_defaults"):
            if _json_bytes(metadata.get(key)) != _json_bytes(reference_meta.get(key)):
                raise ValueError(f"parent {key} mismatch")
        if set(arrays) != keys:
            raise ValueError("parent array key mismatch")
        for name in keys:
            value, baseline = arrays[name], reference[name]
            if value.dtype != baseline.dtype:
                raise ValueError(f"{name}: parent dtype mismatch")
            if name.startswith("mem_"):
                if value.ndim != baseline.ndim or value.shape[1:] != baseline.shape[1:]:
                    raise ValueError(f"{name}: parent memory shape mismatch")
            elif value.shape != baseline.shape:
                raise ValueError(f"{name}: parent shape mismatch")
    count = len(loaded)
    output: dict[str, np.ndarray] = {}
    event_counts = [_validate_memory(arrays) for arrays, _ in loaded]
    event_offsets = np.cumsum([0] + event_counts[:-1], dtype=np.int64).tolist()
    for name in sorted(keys):
        baseline = reference[name]
        values = [arrays[name] for arrays, _ in loaded]
        if name.startswith("fixed_"):
            if any(not np.array_equal(baseline, value) for value in values[1:]):
                raise ValueError(f"{name}: fixed birth arrays differ")
            output[name] = baseline.copy()
        elif name.startswith("plastic_"):
            if baseline.dtype.kind != "f":
                raise ValueError(f"{name}: plastic weights must have floating dtype")
            average = np.zeros(baseline.shape, dtype=np.float64)
            for value in values:
                average += value.astype(np.float64) / count
            output[name] = average.astype(baseline.dtype)
        elif name.startswith("state_"):
            output[name] = np.zeros_like(baseline)
        elif name.startswith("mem_"):
            pieces = []
            for value, offset in zip(values, event_offsets):
                piece = value.copy()
                if name == "mem_event_strength":
                    piece /= count
                elif name == "mem_edge_time":
                    piece += offset
                elif name == "mem_event_next":
                    piece[piece >= 0] += offset
                elif name == "mem_edges":
                    piece[:, 1] += offset
                pieces.append(piece)
            output[name] = np.concatenate(pieces, axis=0)
        else:
            raise ValueError(f"{name}: unknown fusion policy; use fixed_, plastic_, state_, or mem_ arrays")
    output_metadata = {key: reference_meta[key] for key in
                       ("schema_version", "birth_hash", "feature_schema", "memory_protocol")}
    for key in ("model_config", "state_defaults"):
        if key in reference_meta:
            output_metadata[key] = reference_meta[key]
    output_metadata.update({
        "operation": "average_aligned_weights_preserve_all_events",
        "parents": provenance,
        "parent_count": count,
        "memory_strength_scale": 1.0 / count,
        "memory_partitions": [
            {"parent_index": index, "event_start": offset, "event_count": size}
            for index, (offset, size) in enumerate(zip(event_offsets, event_counts))
        ],
    })
    save_checkpoint(destination, output, output_metadata)
    return output_metadata
