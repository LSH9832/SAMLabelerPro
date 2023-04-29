# -*- coding: utf-8 -*-
# @Author  : LG

from segment_anything import sam_model_registry, SamPredictor
import torch
import numpy as np


class SegAny:

    def __init__(self, checkpoint, half=True, force_model_type=None):
        # print(self.model_type)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        success = False

        all_model_type = ["h", "l", "b"]
        if force_model_type is not None and force_model_type in all_model_type:
            all_model_type = [force_model_type]

        for self.model_type in [f"vit_{t}" for t in all_model_type]:
            try:
                half = half and torch.cuda.is_available() and not self.model_type.endswith("h")
                print(f"try load weights '{checkpoint}' with model size '{self.model_type}'")
                sam = sam_model_registry[self.model_type](checkpoint=checkpoint)
                sam.to(device=self.device, dtype=torch.float16 if half else torch.float32)

                print("Use FP16 Precision:", half)
                sam.set_half(True) if half else None
                self.predictor = SamPredictor(sam)
                self.image = None
                success = True
                break
            except Exception as e:
                # print(e)
                pass

        self.success = success

    def set_image(self, image):
        self.predictor.set_image(image)

    def reset_image(self):
        self.predictor.reset_image()
        self.image = None
        torch.cuda.empty_cache()

    def predict(self, input_point, input_label):
        input_point = np.array(input_point)
        input_label = np.array(input_label)

        # print(input_point, input_label)

        masks, scores, logits = self.predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )
        mask_input = logits[np.argmax(scores), :, :]  # Choose the model's best mask
        masks, _, _ = self.predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            mask_input=mask_input[None, :, :],
            multimask_output=False,
        )
        torch.cuda.empty_cache()
        return masks
