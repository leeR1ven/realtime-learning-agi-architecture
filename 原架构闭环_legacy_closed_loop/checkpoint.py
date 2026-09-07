"""Atomic numerical snapshots. No executable or pickled checkpoint objects."""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile

import numpy as np


def _pack(value, arrays):
    if isinstance(value, np.ndarray):
        if value.dtype.kind not in 'biuf' or not np.isfinite(value).all():
            raise ValueError('Snapshot arrays must be finite numeric arrays')
        key = 'array_' + str(len(arrays))
        arrays[key] = value
        return {'__array__': key}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _pack(v, arrays) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_pack(v, arrays) for v in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f'Unsupported snapshot value: {type(value).__name__}')


def _unpack(value, arrays):
    if isinstance(value, dict):
        if set(value) == {'__array__'}:
            return arrays[value['__array__']].copy()
        return {k: _unpack(v, arrays) for k, v in value.items()}
    if isinstance(value, list):
        return [_unpack(v, arrays) for v in value]
    return value


def save_tree(path, tree):
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {}
    meta = _pack(tree, arrays)
    arrays['manifest'] = np.frombuffer(json.dumps(meta, ensure_ascii=False,
        allow_nan=False).encode('utf-8'), np.uint8)
    fd, temporary = tempfile.mkstemp(prefix='.'+path.name, suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def load_tree(path):
    with np.load(Path(path), allow_pickle=False) as arrays:
        tree = json.loads(arrays['manifest'].tobytes().decode('utf-8'))
        return _unpack(tree, arrays)
