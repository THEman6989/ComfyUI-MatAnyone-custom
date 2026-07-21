import importlib.util
import sys
import types
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_NAME = "matanyone_custom_test"

# Load only the modules under test without importing the package's heavyweight
# ComfyUI node registration module.
package = types.ModuleType(PACKAGE_NAME)
package.__path__ = [str(ROOT)]
sys.modules.setdefault(PACKAGE_NAME, package)

comfy = types.ModuleType("comfy")
comfy_utils = types.ModuleType("comfy.utils")
setattr(
    comfy_utils,
    "ProgressBar",
    type(
        "ProgressBar",
        (),
        {"__init__": lambda self, total: None, "update_absolute": lambda *args: None},
    ),
)
setattr(comfy, "utils", comfy_utils)
sys.modules.setdefault("comfy", comfy)
sys.modules.setdefault("comfy.utils", comfy_utils)


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(f"{PACKAGE_NAME}.{name}", ROOT / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


utils = load_module("utils", "utils.py")
mat_anyone2 = load_module("mat_anyone2", "mat_anyone2.py")


def mask_batch():
    masks = torch.zeros(5, 8, 8)
    masks[1, 2:5, 2:5] = 1.0
    masks[3, 1:7, 1:7] = 1.0
    return masks


class MaskGuidanceTests(unittest.TestCase):
    def test_image_input_preserves_mask_batch_dimension(self):
        images = torch.zeros(3, 8, 8, 3)
        images[2, 2:4, 2:4] = 1.0
        masks = utils.get_mask_batch(foreground_mask=images)
        self.assertEqual(masks.shape, (3, 8, 8))
        self.assertEqual(masks[0].max(), 0)
        self.assertGreater(masks[2].max(), 0)

    def test_default_mode_uses_only_first_non_black_mask(self):
        guidance, anchor = mat_anyone2.prepare_mask_guidance(
            mask_batch(), 5, 0, 8, 8, "first_valid_then_propagate"
        )
        self.assertEqual(anchor, 1)
        self.assertEqual(
            [i for i, mask in enumerate(guidance) if mask is not None], [1]
        )

    def test_enhanced_mode_uses_valid_masks_and_skips_black_frames(self):
        guidance, anchor = mat_anyone2.prepare_mask_guidance(
            mask_batch(), 5, 0, 8, 8, "valid_per_frame_then_propagate"
        )
        self.assertEqual(anchor, 1)
        self.assertEqual(
            [i for i, mask in enumerate(guidance) if mask is not None], [1, 3]
        )
        self.assertIsNone(guidance[2])
        self.assertIsNone(guidance[4])

    def test_single_mask_keeps_legacy_mask_frame_placement(self):
        guidance, anchor = mat_anyone2.prepare_mask_guidance(
            torch.ones(1, 8, 8), 6, 4, 8, 8
        )
        self.assertEqual(anchor, 4)
        self.assertIsNotNone(guidance[4])

    def test_fully_black_batch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "No valid mask found"):
            mat_anyone2.prepare_mask_guidance(torch.zeros(4, 8, 8), 4, 0, 8, 8)

    def test_threshold_can_reject_weak_masks(self):
        masks = torch.zeros(2, 8, 8)
        masks[1, 0, 0] = 0.01
        with self.assertRaisesRegex(ValueError, "No valid mask found"):
            mat_anyone2.prepare_mask_guidance(
                masks, 2, 0, 8, 8, mask_valid_threshold=0.02
            )

    def test_per_frame_guidance_is_injected_as_hw_tensor(self):
        class FakeProcessor:
            def __init__(self):
                self.received = None

            def step(self, image, mask=None, objects=None):
                self.received = (mask, objects)
                return torch.zeros(2, 8, 8)

        processor = FakeProcessor()
        mask = torch.ones(8, 8)
        mat_anyone2._processor_step(processor, torch.zeros(3, 8, 8), mask)
        received_mask, objects = processor.received
        self.assertEqual(received_mask.shape, (8, 8))
        self.assertEqual(objects, [1])


if __name__ == "__main__":
    unittest.main()
