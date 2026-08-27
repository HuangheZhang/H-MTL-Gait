"""Coverage checks for every MTL method named in the manuscript."""

import unittest
from pathlib import Path

import torch
import yaml

from scripts.run_mtl_experiments import EXPERIMENT_CONFIGS
from src.models.gradient_methods import create_gradient_method
from src.models.mtl_models import (
    DEFAULT_TASKS,
    MTL_MODEL_REGISTRY,
    create_mtl_model,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
METHOD_CONFIG = REPO_ROOT / "configs" / "paper_mtl_methods.yaml"

EXPECTED_EXPERIMENTS = {
    "hard_sharing": ("hard_sharing", "none", {}),
    "mmoe": ("mmoe", "none", {}),
    "ple": ("ple", "none", {}),
    "pcgrad": ("hard_sharing", "pcgrad", {}),
    "cagrad": ("hard_sharing", "cagrad", {"c": 0.4}),
    "uncertainty_weighting": (
        "hard_sharing", "uncertainty", {"num_tasks": 10}
    ),
    "dwa": (
        "hard_sharing", "dwa", {"num_tasks": 10, "temperature": 2.0}
    ),
    "mmoe_dwa": (
        "mmoe", "dwa", {"num_tasks": 10, "temperature": 2.0}
    ),
    "mmoe_cagrad": ("mmoe", "cagrad", {"c": 0.4}),
}

EXPECTED_OUTPUT_SHAPES = {
    task: (2, config["num_classes"] if config["type"] == "classification" else 1)
    for task, config in DEFAULT_TASKS.items()
}


class PaperMethodConfigurationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with METHOD_CONFIG.open("r", encoding="utf-8") as stream:
            cls.config = yaml.safe_load(stream)

    def test_yaml_and_runner_mappings_cover_required_ids(self):
        methods = self.config["methods"]
        self.assertTrue(EXPECTED_EXPERIMENTS.keys() <= methods.keys())
        self.assertTrue(EXPECTED_EXPERIMENTS.keys() <= EXPERIMENT_CONFIGS.keys())
        for experiment_id, (model_type, gradient_method, kwargs) in (
            EXPECTED_EXPERIMENTS.items()
        ):
            with self.subTest(experiment_id=experiment_id):
                public_config = methods[experiment_id]
                runner_config = EXPERIMENT_CONFIGS[experiment_id]
                self.assertEqual(public_config["model_type"], model_type)
                self.assertEqual(public_config["gradient_method"], gradient_method)
                self.assertEqual(public_config["gradient_kwargs"], kwargs)
                self.assertEqual(runner_config["model_type"], model_type)
                self.assertEqual(runner_config["gradient_method"], gradient_method)
                self.assertEqual(runner_config.get("gradient_kwargs", {}), kwargs)
                self.assertEqual(
                    public_config["dimensions"]["base_model"], model_type
                )
                self.assertEqual(
                    public_config["dimensions"]["optimization_strategy"],
                    gradient_method,
                )

    def test_distinct_paper_base_models_forward_all_tasks(self):
        self.assertTrue({"hard_sharing", "mmoe", "ple"} <= MTL_MODEL_REGISTRY.keys())
        model_kwargs = {
            "hard_sharing": {
                "encoder_type": "cnn1d",
                "channels": [8],
                "kernel_sizes": [3],
                "head_hidden_dim": 8,
                "dropout": 0.0,
            },
            "mmoe": {
                "num_experts": 2,
                "expert_hidden": 8,
                "expert_output_dim": 8,
                "head_hidden_dim": 8,
                "dropout": 0.0,
            },
            "ple": {
                "encoder_type": "cnn1d",
                "channels": [8],
                "kernel_sizes": [3],
                "num_extraction_layers": 1,
                "num_task_experts": 1,
                "num_shared_experts": 1,
                "expert_hidden": 8,
                "head_hidden_dim": 8,
                "dropout": 0.0,
            },
        }
        inputs = torch.zeros(2, 200, 36)
        for model_type, kwargs in model_kwargs.items():
            with self.subTest(model_type=model_type):
                model = create_mtl_model(
                    model_type=model_type,
                    input_channels=36,
                    tasks=DEFAULT_TASKS,
                    **kwargs,
                ).cpu().eval()
                with torch.inference_mode():
                    outputs = model(inputs)
                self.assertEqual(set(outputs), set(DEFAULT_TASKS))
                self.assertEqual(
                    {name: tuple(value.shape) for name, value in outputs.items()},
                    EXPECTED_OUTPUT_SHAPES,
                )

    def test_paper_optimization_factories(self):
        method_specs = {
            "none": ({}, {}),
            "pcgrad": ({}, {}),
            "cagrad": ({"c": 0.4}, {"c": 0.4}),
            "uncertainty": ({"num_tasks": 10}, {"num_tasks": 10}),
            "dwa": (
                {"num_tasks": 10, "temperature": 2.0},
                {"num_tasks": 10, "temperature": 2.0},
            ),
        }
        for method_name, (kwargs, expected_attributes) in method_specs.items():
            with self.subTest(method_name=method_name):
                method = create_gradient_method(method_name, **kwargs)
                self.assertEqual(method.name, method_name)
                for attribute, expected_value in expected_attributes.items():
                    self.assertEqual(getattr(method, attribute), expected_value)


if __name__ == "__main__":
    unittest.main()
