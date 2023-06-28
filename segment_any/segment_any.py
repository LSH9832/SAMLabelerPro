from segment_anything import sam_model_registry, SamPredictor
import torch
import numpy as np


class SegAny:

    def __init__(self, checkpoint, half=True, force_model_type=None):
        # print(self.model_type)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        success = False

        all_model_type = ["mobile", "h", "l", "b"]
        if force_model_type is not None and force_model_type in all_model_type:
            all_model_type = [force_model_type]

        for self.model_type in [f"vit_{t}" for t in all_model_type]:
            try:
                half = half and torch.cuda.is_available() and not self.model_type.endswith("h")
                if (self.model_type.endswith("h") or self.model_type.endswith("mobile")) and half:
                    print(f"{self.model_type} can not run with half precision, using full precision.")
                    half = False
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
        # print("set image")
        self.predictor.set_image(image)
        # print("done")

    def reset_image(self):
        self.predictor.reset_image()
        self.image = None
        torch.cuda.empty_cache()

    def predict_box(self, box, xyxy=True, expand=0):

        def modify_box(bbox: (list, np.ndarray), xy2=True):
            if isinstance(bbox, list):
                bbox = np.array(bbox)
            if not xy2:
                bbox[2:4] += bbox[:2]
            return bbox

        box = modify_box(box, xyxy)
        ori_box = box.copy()
        box[:2] -= expand
        box[2:4] += expand
        masks, scores, logits = self.predictor.predict(box=box, multimask_output=True)
        mask_input = logits[np.argmax(scores), :, :]  # Choose the model's best mask
        masks, _, mi = self.predictor.predict(
            box=box,
            mask_input=mask_input[None, :, :],
            multimask_output=False,
        )

        modify_mask: np.ndarray = masks.copy()
        modify_mask[..., ori_box[1]:ori_box[3], ori_box[0]:ori_box[2]] = False
        masks[modify_mask] = False

        torch.cuda.empty_cache()
        return masks


    def predict(self, input_point, input_label, box=None):
        input_point = np.array(input_point)
        input_label = np.array(input_label)
        if isinstance(box, list):
            box = np.array(box)

        # print(input_point, input_label)

        masks, scores, logits = self.predictor.predict(
            box=box,
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )
        mask_input = logits[np.argmax(scores), :, :]  # Choose the model's best mask
        masks, _, _ = self.predictor.predict(
            box=box,
            point_coords=input_point,
            point_labels=input_label,
            mask_input=mask_input[None, :, :],
            multimask_output=False,
        )
        torch.cuda.empty_cache()
        return masks
