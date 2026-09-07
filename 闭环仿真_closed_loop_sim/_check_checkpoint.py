"""Focused integrity checks; all artifacts live in a temporary directory."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import checkpoint


class CheckpointChecks(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)
        self.metadata = {
            "birth_hash": "test-common-birth",
            "feature_schema": {"neurons": 8, "audio": [0, 1]},
            "model_config": {"decay": 0.001},
            "state_defaults": {"pfc_inhibition": 0.2},
            "frame": 23,
        }
        self.arrays = {
            "fixed_projection": np.eye(2, dtype=np.float32),
            "plastic_pfc": np.full((2, 2), 2.0, dtype=np.float32),
            "plastic_audio": np.array([[1, 3], [5, 7]], dtype=np.float32),
            "state_pfc_inhibition": np.full(2, 0.3, dtype=np.float32),
            "state_tick": np.array(23, dtype=np.int64),
            "mem_event_tick": np.array([0, 1], dtype=np.int64),
            "mem_event_strength": np.array([0.8, 0.6], dtype=np.float32),
            "mem_event_next": np.array([1, -1], dtype=np.int64),
            "mem_event_action": np.array([[1, 0], [0, 1]], dtype=np.float32),
            "mem_edge_source": np.array([0, 8, 1, 9], dtype=np.int64),
            "mem_edge_time": np.array([0, 0, 1, 1], dtype=np.int64),
            "mem_edge_weight": np.array([1, 0.9, 1, 0.7], dtype=np.float32),
        }

    def tearDown(self):
        self.temporary.cleanup()

    def save(self, name, arrays=None, metadata=None):
        path = self.directory / name
        checkpoint.save_checkpoint(path, self.arrays if arrays is None else arrays,
                                   self.metadata if metadata is None else metadata)
        return path

    def test_roundtrip_and_extension(self):
        path = self.save("model.snapshot")
        arrays, metadata = checkpoint.load_checkpoint(path)
        self.assertEqual(set(arrays), set(self.arrays))
        for name in arrays:
            np.testing.assert_array_equal(arrays[name], self.arrays[name], strict=True)
        self.assertEqual(metadata["schema_version"], 1)
        self.assertEqual(metadata["frame"], 23)
        self.assertEqual(list(self.directory.iterdir()), [path])

    def test_interrupted_write_preserves_last_checkpoint(self):
        path = self.save("model.npz")
        original = path.read_bytes()

        def fail(handle, **kwargs):
            handle.write(b"partial checkpoint")
            raise OSError("simulated full disk")

        with patch.object(checkpoint.np, "savez", fail):
            with self.assertRaises(OSError):
                self.save("model.npz")
        self.assertEqual(path.read_bytes(), original)
        self.assertEqual(list(self.directory.iterdir()), [path])
        checkpoint.load_checkpoint(path)

    def test_average_and_keep_all_memory_events(self):
        first = self.save("first.npz")
        second_arrays = {key: value.copy() for key, value in self.arrays.items()}
        second_arrays["plastic_pfc"][:] = 6
        second_arrays["mem_event_next"][:] = [1, 0]  # The second memory is a ring.
        second = self.save("second.npz", second_arrays)
        before = [path.read_bytes() for path in (first, second)]
        out = self.directory / "fusion.npz"
        returned = checkpoint.merge_checkpoints([first, second], out)
        arrays, metadata = checkpoint.load_checkpoint(out)
        np.testing.assert_array_equal(arrays["plastic_pfc"], np.full((2, 2), 4))
        np.testing.assert_array_equal(arrays["fixed_projection"], self.arrays["fixed_projection"])
        np.testing.assert_array_equal(arrays["mem_event_tick"], [0, 1, 0, 1])
        np.testing.assert_array_equal(arrays["mem_event_next"], [1, -1, 3, 2])
        np.testing.assert_array_equal(arrays["mem_edge_time"], [0, 0, 1, 1, 2, 2, 3, 3])
        np.testing.assert_array_equal(arrays["mem_event_strength"],
                                      np.array([0.4, 0.3, 0.4, 0.3], dtype=np.float32))
        np.testing.assert_array_equal(arrays["mem_edge_weight"],
                                      np.tile(self.arrays["mem_edge_weight"], 2))
        self.assertTrue((arrays["state_pfc_inhibition"] == 0).all())
        self.assertEqual(arrays["state_tick"], 0)
        self.assertNotIn("frame", metadata)
        self.assertEqual(metadata["model_config"], self.metadata["model_config"])
        self.assertEqual(metadata["state_defaults"], self.metadata["state_defaults"])
        self.assertEqual(len(metadata["parents"]), 2)
        self.assertEqual(len(metadata["parents"][0]["sha256"]), 64)
        self.assertEqual(metadata, returned)
        self.assertEqual([path.read_bytes() for path in (first, second)], before)

    def test_variable_event_counts_and_empty_memory(self):
        first = self.save("first.npz")
        empty = {key: value[:0].copy() if key.startswith("mem_") else value.copy()
                 for key, value in self.arrays.items()}
        second = self.save("empty.npz", empty)
        out = self.directory / "fusion.npz"
        checkpoint.merge_checkpoints([second, first], out)
        arrays, metadata = checkpoint.load_checkpoint(out)
        self.assertEqual(len(arrays["mem_event_tick"]), 2)
        np.testing.assert_array_equal(arrays["mem_event_next"], [1, -1])
        self.assertEqual(metadata["memory_partitions"][0]["event_count"], 0)

    def test_incompatible_birth_schema_shapes_dtypes_fixed_and_keys_rejected(self):
        first = self.save("first.npz")
        cases = []
        for field, value in (("birth_hash", "other"), ("feature_schema", {"neurons": 9}),
                             ("model_config", {"decay": 1})):
            meta = dict(self.metadata)
            meta[field] = value
            cases.append((self.arrays, meta))
        for replacement in (np.zeros((3, 3), np.float32), np.zeros((2, 2), np.float64)):
            arrays = dict(self.arrays)
            arrays["plastic_pfc"] = replacement
            cases.append((arrays, self.metadata))
        arrays = dict(self.arrays)
        arrays["fixed_projection"] = np.zeros((2, 2), np.float32)
        cases.append((arrays, self.metadata))
        arrays = dict(self.arrays)
        arrays.pop("plastic_audio")
        cases.append((arrays, self.metadata))
        for index, (arrays, metadata) in enumerate(cases):
            with self.subTest(index=index):
                second = self.save(f"bad-{index}.npz", arrays, metadata)
                output = self.directory / f"rejected-{index}.npz"
                with self.assertRaises(ValueError):
                    checkpoint.merge_checkpoints([first, second], output)
                self.assertFalse(output.exists())

    def test_parent_overwrite_and_duplicate_paths_rejected(self):
        first = self.save("first.npz")
        before = first.read_bytes()
        with self.assertRaises(ValueError):
            checkpoint.merge_checkpoints([first], first)
        with self.assertRaises(ValueError):
            checkpoint.merge_checkpoints([first, first], self.directory / "fusion.npz")
        self.assertEqual(first.read_bytes(), before)

    def test_pickle_nonfinite_and_invalid_event_rejected(self):
        for name, value in (("plastic_pfc", np.array([np.nan])),
                            ("state_bad", np.array([object()], dtype=object)),
                            ("mem_edge_time", np.array([0, 0, 1, 2], np.int64))):
            arrays = dict(self.arrays)
            arrays[name] = value
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.save("bad.npz", arrays)
        valid = self.save("valid.npz")
        with np.load(valid, allow_pickle=False) as archive:
            values = {key: archive[key] for key in archive.files}
        values["state_bad"] = np.array([object()], dtype=object)
        hostile = self.directory / "object-array.npz"
        np.savez(hostile, **values)
        with self.assertRaises(ValueError):
            checkpoint.load_checkpoint(hostile)

    def test_packed_edges_offset_without_deduplication(self):
        arrays = {key: value for key, value in self.arrays.items() if not key.startswith("mem_edge_")}
        arrays["mem_edges"] = np.array([[0, 0, 1], [8, 0, .9], [1, 1, 1], [9, 1, .7]], np.float64)
        first = self.save("packed-first.npz", arrays)
        second = self.save("packed-second.npz", arrays)
        output = self.directory / "packed-fusion.npz"
        checkpoint.merge_checkpoints([first, second], output)
        fused, _ = checkpoint.load_checkpoint(output)
        self.assertEqual(fused["mem_edges"].shape, (8, 3))
        np.testing.assert_array_equal(fused["mem_edges"][:, 1], [0, 0, 1, 1, 2, 2, 3, 3])


if __name__ == "__main__":
    unittest.main(verbosity=2)
