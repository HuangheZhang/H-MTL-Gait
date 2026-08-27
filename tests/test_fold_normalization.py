"""Regression tests for training-fold-only channel standardization."""

import copy
import io
import json
import unittest
from unittest import mock

import numpy as np
import pandas as pd
import torch

from src.data.dataset import GaitDataLoader, GaitDataset
from src.data.normalization import FoldStandardizer
from src.trainers.mtl_trainer_v2 import MTLCrossValidator


class FoldStandardizerTests(unittest.TestCase):
    def setUp(self):
        self.train = np.array(
            [
                [[1.0, 10.0], [3.0, np.nan], [5.0, 30.0]],
                [[7.0, 40.0], [9.0, np.inf], [11.0, 60.0]],
            ],
            dtype=np.float64,
        )
        self.held_out = np.full((3, 3, 2), 10000.0, dtype=np.float64)
        self.held_out[0, 0, 0] = -np.inf

    def test_fit_uses_training_values_only(self):
        standardizer = FoldStandardizer().fit(self.train)
        expected_clean = np.nan_to_num(
            self.train, nan=0.0, posinf=0.0, neginf=0.0
        )
        expected_mean = expected_clean.reshape(-1, 2).mean(axis=0)
        np.testing.assert_allclose(standardizer.state_dict()["mean"], expected_mean)

        state_before = copy.deepcopy(standardizer.state_dict())
        transformed_held_out = standardizer.transform(self.held_out)
        self.assertEqual(state_before, standardizer.state_dict())
        self.assertGreater(abs(float(transformed_held_out.mean())), 100.0)

    def test_shapes_and_nonfinite_values(self):
        standardizer = FoldStandardizer()
        transformed_train = standardizer.fit_transform(self.train)
        transformed_held_out = standardizer.transform(self.held_out)
        self.assertEqual(transformed_train.shape, self.train.shape)
        self.assertEqual(transformed_held_out.shape, self.held_out.shape)
        self.assertEqual(transformed_train.dtype, np.float32)
        self.assertTrue(np.isfinite(transformed_train).all())
        self.assertTrue(np.isfinite(transformed_held_out).all())
        np.testing.assert_allclose(
            transformed_train.reshape(-1, 2).mean(axis=0),
            np.zeros(2),
            atol=1e-6,
        )

    def test_state_scope_count_and_serialization(self):
        state = FoldStandardizer().fit(self.train).state_dict()
        self.assertEqual(state["scope"], "training_fold_only")
        self.assertEqual(state["channels"], 2)
        self.assertEqual(state["fit_sample_count"], 6)
        json.dumps(state)
        buffer = io.BytesIO()
        torch.save(state, buffer)
        self.assertGreater(buffer.tell(), 0)

    def test_state_round_trip(self):
        fitted = FoldStandardizer().fit(self.train)
        restored = FoldStandardizer().load_state_dict(fitted.state_dict())
        np.testing.assert_allclose(
            fitted.transform(self.held_out), restored.transform(self.held_out)
        )


class FoldPipelineIntegrationTests(unittest.TestCase):
    def test_cross_validator_uses_only_explicit_training_indices(self):
        train = np.array(
            [
                [[1.0, 10.0], [3.0, 20.0], [5.0, 30.0]],
                [[7.0, 40.0], [9.0, 50.0], [11.0, 60.0]],
            ],
            dtype=np.float32,
        )
        held_out = np.full((4, 3, 2), 10000.0, dtype=np.float32)
        data = np.concatenate([train, held_out], axis=0)
        labels = {"fine": np.arange(6, dtype=np.int64)}
        metadata = pd.DataFrame({
            "subject": [f"S{index}" for index in range(6)],
            "pathology": ["HS", "HS", "CVA", "CVA", "PD", "PD"],
        })
        dataset = GaitDataset(data, labels, metadata)
        validator = MTLCrossValidator(
            model_factory=lambda: None,
            optimizer_factory=lambda model: None,
            normalize_inputs=True,
        )

        train_dataset, val_dataset, test_dataset, state = (
            validator._make_fold_datasets(
                dataset, metadata, train_indices=[0, 1],
                val_indices=[2, 3], test_indices=[4, 5]
            )
        )

        expected_mean = train.reshape(-1, 2).mean(axis=0)
        np.testing.assert_allclose(state["mean"], expected_mean)
        np.testing.assert_allclose(
            train_dataset.data.numpy().reshape(-1, 2).mean(axis=0),
            np.zeros(2), atol=1e-6
        )
        self.assertEqual(state["scope"], "training_fold_only")
        self.assertEqual(state["channels"], 2)
        self.assertEqual(state["fit_sample_count"], 6)
        self.assertGreater(float(val_dataset.data.mean()), 100.0)
        self.assertGreater(float(test_dataset.data.mean()), 100.0)

    def test_prepare_dataset_rejects_normalization_before_io(self):
        loader = GaitDataLoader("missing-data-root")
        with mock.patch.object(
            loader, "load_all_data", side_effect=AssertionError("I/O attempted")
        ) as load_all_data:
            with self.assertRaisesRegex(ValueError, "after subject splitting"):
                loader.prepare_dataset(normalize=True, verbose=False)
        load_all_data.assert_not_called()


if __name__ == "__main__":
    unittest.main()
