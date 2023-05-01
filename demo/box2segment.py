import utils.datasets as dataset
from segment_any.segment_any import SegAny

from glob import glob
import os.path as osp
import os
import numpy as np
import cv2
from annotation import Annotation, Object


ROOT_PATH = osp.dirname(osp.dirname(__file__))


def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--cfg", type=str, default="cfg/dataset/visdrone.yaml", help="dataset config file")
    parser.add_argument("--size", type=str, default="h", help="model size")
    parser.add_argument("--half", action="store_true", help="use half precision")
    parser.add_argument("--dist", type=str, default="datasets/visdrone/annotations", help="path to save annotations")
    parser.add_argument("--val", action="store_true", help="run in val dataset, or train dataset")
    parser.add_argument("--overwrite", action="store_true", help="overwrite exist annotations")
    return parser.parse_args()


class SegBox:

    model = None
    image = None

    def __init__(self, half=True, force_size=None):
        self.load_SAM_model(half, force_size)

    def set_image(self, image):
        if self.model is not None:
            self.image = image.copy()
            self.model.set_image(image.copy())

    def load_SAM_model(self, half=True, force_size=None):
        weights_list = glob(osp.join(ROOT_PATH, "**/*.pth").replace("\\", "/"), recursive=True)
        weights_size = [os.path.getsize(file) / 1024 ** 3 for file in weights_list]
        for _, file in sorted(zip(weights_size, weights_list), reverse=True):
            try:
                file = file.replace("\\", "/")
                SAM_model = SegAny(file, half, force_size)
                if SAM_model.success:
                    print(f'Using weights: {file}.')
                    self.model = SAM_model
                    return
            except:
                pass
        return None

    def seg_one_box(self, box, xyxy, expand=0):
        if self.model is None:
            return
        return self.model.predict_box(box, xyxy, expand)

    def draw_mask(self, mask, image=None, color=(0, 255, 0), ratio=0.5):
        if image is None:
            image = self.image.copy()
        assert image is not None
        image = image.astype(float)
        image[mask] *= (1. - ratio)
        image[mask] += np.array([color]) * ratio
        return image.astype("uint8")


def main(args):
    assert args.size.lower() in ["b", "l", "h"]
    boxSegmenter = SegBox(force_size=args.size, half=args.half)

    if not (args.cfg.startswith("/") or args.cfg[1] == ":"):
        args.cfg = osp.join(ROOT_PATH, args.cfg).replace("\\", "/")

    args.dist = osp.join(args.dist, "val" if args.val else "train").replace("\\", "/")
    if not (args.dist.startswith("/") or args.dist[1] == ":"):
        args.dist = osp.join(ROOT_PATH, args.dist).replace("\\", "/")

    my_dataset: dataset.VisDroneDataset = dataset.get_dataset(args.cfg, mode="val" if args.val else "train")

    num_imgs = len(my_dataset)

    if num_imgs:
        os.makedirs(args.dist, exist_ok=True)


    print()
    for idx in range(num_imgs):


        if isinstance(my_dataset, dataset.COCODataset):
            image_name = my_dataset.annotations[idx][3]
            image_name = os.path.join(my_dataset.train_dir, image_name).replace('\\', '/')
        else:
            image_name: str = my_dataset.annotation_list[idx]["image"]

        name = osp.basename(image_name).split(".")[0]
        label_name = osp.join(args.dist, name + ".json").replace("\\", "/")

        if osp.isfile(label_name) and not args.overwrite:
            continue

        image, bboxes, image_name = my_dataset.pull_origin_item(idx)
        bboxes, classes = bboxes[..., :4].astype(int), bboxes[..., 4].astype(int)
        label = Annotation(image_name, label_name, image.copy())

        boxSegmenter.set_image(image)

        for group_id, (box, cls) in enumerate(zip(bboxes, classes)):
            # image_show = cv2.rectangle(image.copy(), (box[0], box[1]), (box[2], box[3]), (0,255,0), 2, cv2.LINE_AA)

            mask = boxSegmenter.seg_one_box(box, xyxy=True, expand=0)[0]
            points, _ = cv2.findContours(mask.astype("uint8"), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)

            if len(points):
                areas = []
                for edge in points:
                    this_mask = np.zeros_like(mask, dtype="uint8")
                    # print(len(edge), edge.shape, edge.transpose(1, 0, 2).shape)
                    cv2.fillPoly(this_mask, edge.transpose(1, 0, 2), 1)
                    areas.append(np.sum(this_mask))

                max_area = max(areas)
                for area, edge in zip(areas, points):
                    if area * 20 > max_area:
                        label.objects.append(Object(
                            category=my_dataset.names[cls],
                            group=group_id+1,
                            segmentation=np.round(edge.transpose(1, 0, 2)[0]).astype(int).tolist(),
                            area=int(area),
                            layer=group_id+1,
                            bbox=box.astype(int).tolist()
                        ))

            else:
                x1, y1, x2, y2 = [int(loc) for loc in box]
                label.objects.append(Object(
                    category=my_dataset.names[cls],
                    group=group_id + 1,
                    segmentation=[[x1, y1], [x1, y2], [x2, y2], [x2, y1]],
                    area=int((box[2] - box[0]) * (box[3] - box[1])),
                    layer=group_id + 1,
                    bbox=box.astype(int).tolist()
                ))


        label.save_annotation()
        print(f"\r{idx + 1}/{num_imgs}: {image_name}", end="")
    print()


if __name__ == '__main__':
    main(get_args())
