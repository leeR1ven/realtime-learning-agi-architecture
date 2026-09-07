"""Efficient deterministic hashes of complete route experiment object states.

``digest((environment, controller))`` includes arrays, sparse posting indexes,
cached values, random generators and object attributes. It neither changes nor
serializes the live model. Integer posting sets and integer sequences use one
canonical int64 byte block instead of a separate SHA-256 for every integer.
Hashes are not intended to match the older audit helper's byte format.
"""
from __future__ import annotations

from collections import deque
import hashlib
import struct

import numpy as np


_FORMAT = b"route-complete-state-digest-v1\0"


def digest(value) -> str:
    """Return a SHA-256 of values, types and complete supported object state.

    Dictionary/set insertion order is immaterial. Lists and deques keep order;
    deque capacity, array dtype/shape, object class and RNG state are included.
    Repeated references are compared by value; cycles use an ancestor marker.
    Unsupported values raise TypeError rather than silently omit state.
    """
    hasher = hashlib.sha256(_FORMAT)
    active = {}

    def block(data):
        hasher.update(struct.pack("<Q", len(data)))
        hasher.update(data)

    def integers(items):
        # Only exact Python ints qualify: bool and numpy scalar types retain
        # their own type tags through the general path. Overflow falls back
        # without losing Python's arbitrary-precision integer values.
        if not all(type(item) is int for item in items):
            return False
        try:
            packed = np.asarray(items, dtype="<i8")
        except (OverflowError, ValueError):
            return False
        hasher.update(b"integer-block\0")
        block(memoryview(packed).cast("B"))
        return True

    def attributes(item):
        if hasattr(item, "__dict__"):
            add(vars(item))
        # Include inherited slots as well as the ordinary instance dictionary.
        slots = {}
        for owner in type(item).__mro__:
            names = owner.__dict__.get("__slots__", ())
            names = (names,) if isinstance(names, str) else names
            for name in names:
                if name in ("__dict__", "__weakref__"):
                    continue
                attr = name
                if name.startswith("__") and not name.endswith("__"):
                    attr = "_" + owner.__name__.lstrip("_") + name
                if hasattr(item, attr):
                    slots[owner.__module__ + "." + owner.__qualname__ + ":" + name] = getattr(item, attr)
        if slots:
            add(slots)

    def add(item):
        label = (type(item).__module__ + "." + type(item).__qualname__).encode("utf-8")
        block(label)
        scalar = item is None or isinstance(item, (str, bytes, int, float, bool, complex, np.generic))
        identity = id(item)
        if not scalar:
            if identity in active:
                hasher.update(b"ancestor\0")
                hasher.update(struct.pack("<Q", active[identity]))
                return
            active[identity] = len(active)
        try:
            if isinstance(item, np.ndarray):
                block(item.dtype.str.encode())
                # dtype.str collapses structured field names/layout; include
                # the descriptor too so those changes cannot go unnoticed.
                if item.dtype.fields is not None:
                    add(item.dtype.descr)
                add(item.shape)
                if item.dtype.hasobject:
                    for element in item.flat:
                        add(element)
                else:
                    contiguous = np.ascontiguousarray(item)
                    block(memoryview(contiguous.reshape(-1)).cast("B"))
            elif isinstance(item, np.generic):
                add(item.item())
            elif isinstance(item, np.random.Generator):
                add(item.bit_generator.state)
            elif isinstance(item, np.random.RandomState):
                add(item.get_state())
            elif isinstance(item, np.random.BitGenerator):
                add(item.state)
            elif isinstance(item, dict):
                # event_slot_by_id is a large integer-to-integer dictionary.
                # Bulk encoding avoids rehashing each key and value separately.
                if item and all(type(k) is int and type(v) is int for k, v in item.items()):
                    ordered = sorted(item.items())
                    try:
                        packed = np.asarray(ordered, dtype="<i8")
                    except (OverflowError, ValueError):
                        packed = None
                    if packed is not None:
                        hasher.update(b"integer-map\0")
                        block(memoryview(packed).cast("B"))
                        if type(item) is not dict:
                            attributes(item)
                        return
                hasher.update(b"mapping\0")
                # Sorting child hashes also supports nested hashable keys;
                # repr() is not used because it may contain object addresses.
                keys = sorted(item, key=digest)
                hasher.update(struct.pack("<Q", len(keys)))
                for key in keys:
                    add(key)
                    add(item[key])
            elif isinstance(item, (set, frozenset)):
                if all(type(child) is int for child in item) and integers(sorted(item)):
                    if type(item) not in (set, frozenset):
                        attributes(item)
                    return
                hasher.update(b"unordered\0")
                children = sorted(digest(child) for child in item)
                hasher.update(struct.pack("<Q", len(children)))
                for child in children:
                    block(child.encode("ascii"))
            elif isinstance(item, (list, tuple, deque)):
                if isinstance(item, deque):
                    add(item.maxlen)
                sequence = list(item) if isinstance(item, deque) else item
                if not integers(sequence):
                    hasher.update(b"sequence\0")
                    hasher.update(struct.pack("<Q", len(sequence)))
                    for child in sequence:
                        add(child)
            elif item is None:
                hasher.update(b"none\0")
            elif isinstance(item, bool):
                hasher.update(b"true\0" if item else b"false\0")
            elif isinstance(item, int):
                block(str(item).encode("ascii"))
            elif isinstance(item, float):
                hasher.update(struct.pack("<d", item))
            elif isinstance(item, complex):
                hasher.update(struct.pack("<dd", item.real, item.imag))
            elif isinstance(item, str):
                block(item.encode("utf-8"))
            elif isinstance(item, bytes):
                block(item)
            elif hasattr(item, "__dict__") or any("__slots__" in cls.__dict__ for cls in type(item).__mro__):
                attributes(item)
                return
            else:
                raise TypeError(f"Unsupported state type: {type(item)!r}")
            # Container subclasses can carry additional instance attributes.
            if hasattr(item, "__dict__") or (
                    isinstance(item, (dict, set, frozenset, list, tuple, deque))
                    and type(item) not in (dict, set, frozenset, list, tuple, deque)):
                attributes(item)
        finally:
            if not scalar:
                active.pop(identity, None)

    add(value)
    return hasher.hexdigest()


state_digest = digest


def _self_check():
    import copy
    import time

    class Fixture:
        pass

    fixture = Fixture()
    fixture.array = np.arange(15, dtype=np.float32).reshape(3, 5)
    fixture.mapping = {"a": 1, "nested": {9: 2, 4: 3}}
    fixture.postings = [{9, 4, 1}, set()]
    fixture.sequence = [1, 3, 2]
    fixture.trace = deque([(1, 2), (3, 4)], maxlen=12)
    fixture.rng = np.random.default_rng(831)
    original = digest(fixture)
    assert digest(copy.deepcopy(fixture)) == original
    for mutate in (
        lambda x: x.array.__setitem__((0, 0), 99),
        lambda x: x.mapping.__setitem__("a", 2),
        lambda x: x.postings[0].add(7),
        lambda x: x.sequence.reverse(),
        lambda x: setattr(x, "trace", deque(x.trace, maxlen=13)),
        lambda x: x.rng.random(),
        lambda x: setattr(x, "new_attribute", True),
    ):
        changed = copy.deepcopy(fixture)
        mutate(changed)
        assert digest(changed) != original
    assert digest({"a": 1, "b": 2}) == digest({"b": 2, "a": 1})
    assert digest({2, 1, 3}) == digest({3, 2, 1})
    assert digest([1, 2]) != digest([True, 2])
    assert digest(np.array([1], dtype="i4")) != digest(np.array([1], dtype="i8"))
    assert digest(np.zeros((2, 3))) != digest(np.zeros((3, 2)))
    assert digest(np.zeros((0, 3))) != digest(np.zeros((3, 0)))
    assert digest([2**100]) == digest(copy.deepcopy([2**100]))
    cycle = []
    cycle.append(cycle)
    assert digest(cycle) == digest(copy.deepcopy(cycle))
    million = [set(range(j, j + 1000)) for j in range(1000)]
    start = time.perf_counter()
    large_hash = digest(million)
    elapsed = time.perf_counter() - start
    assert large_hash == digest(copy.deepcopy(million))
    return {"deepcopy_equal": True, "seven_independent_mutations_detected": True,
            "order_type_dtype_shape_bigint_cycle_checks": True,
            "integer_postings": 1_000_000, "hash_seconds": elapsed}


if __name__ == "__main__":
    import json
    print(json.dumps(_self_check(), indent=2))
