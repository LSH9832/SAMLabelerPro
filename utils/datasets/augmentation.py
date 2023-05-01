from .datasets_wrapper import Dataset
from .basedataset import BaseDataset
from .mask_coding import encode_mask

from ..data_augment import random_affine

import random
import math
import numpy as np
import cv2
from copy import deepcopy


# function for label filter---------------------------------------------------------------------------------------------
# count rest area via bounding box
def count_area(labels):
    return (labels[:, 2] - labels[:, 0]) * (labels[:, 3] - labels[:, 1])


# count rest area via segmentation
def count_area_mask(segments):
    areas = []
    for obj in segments:
        this_area = 0
        for edge in obj:
            edge = np.round(edge).astype(int)
            edge -= np.min(edge)
            try:
                im = np.zeros([np.max(edge)] * 2, dtype="uint8")
            except ValueError:
                continue

            # edge = np.concatenate([edge, [edge[0]]])

            # print(edge)
            this_area += np.sum(np.greater(cv2.fillPoly(im, np.array([edge]), 1), 0))
        areas.append(this_area)

    return np.array(areas)


# function for augmentation---------------------------------------------------------------------------------------------
# get random params in range
def get_aug_params(value, center=None):
    if isinstance(value, (float, int)):
        if center is None:
            center = 0.0
        return random.uniform(center - value, center + value)
    elif isinstance(value, (tuple, list, np.ndarray)):
        if center is None or center <= min(value) or center >= max(value):
            return random.uniform(min(value), max(value))
        else:
            ab = (min(value), center) if random.random() > 0.5 else (center, max(value))
            return random.uniform(*ab)
    else:
        raise ValueError


# get affine transform matrix(rotate, translate and shear)
def get_affine_matrix(
    current_size,
    degrees=10,
    translate=0.1,
    scales=(0.1, 2),
    shear=10.,
):
    twidth, theight = current_size

    # Rotation and Scale
    angle = get_aug_params(degrees)
    scale = get_aug_params(scales, center=1.0)

    if scale <= 0.0:
        raise ValueError("Argument scale should be positive")

    R = cv2.getRotationMatrix2D(angle=angle, center=(int(twidth / 2), int(theight / 2)), scale=scale)

    M = np.ones([2, 3])
    # Shear
    shear_x = math.tan(get_aug_params(shear) * math.pi / 180)
    shear_y = math.tan(get_aug_params(shear) * math.pi / 180)

    M[0] = R[0] + shear_y * R[1]
    M[1] = R[1] + shear_x * R[0]

    # Translation
    translation_x = (get_aug_params(translate) + 0.5 * (1 - scale)) * twidth  # x translation (pixels)
    translation_y = (get_aug_params(translate) + 0.5 * (1 - scale)) * theight  # y translation (pixels)

    M[0, 2] = translation_x
    M[1, 2] = translation_y

    return M, scale


def apply_affine_to_bboxes(targets, target_size, M, scale, segms=None):

    def trans_points(points):       # apply affine transform
        points_c = np.ones([len(points), 3])
        points_c[..., :2] = points
        return points_c @ M.T

    twidth, theight = target_size
    # print(targets)
    if segms is None:
        num_gts = len(targets)

        # warp corner points

        corner_points = targets[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(4 * num_gts, 2)
        corner_points = trans_points(corner_points)

        # print(corner_points)

        corner_points = corner_points.reshape(num_gts, 8)
        # print(corner_points)
        # time.sleep(1000)

        # create new boxes
        corner_xs = corner_points[:, 0::2]
        corner_ys = corner_points[:, 1::2]
        new_bboxes = (
            np.concatenate(
                (corner_xs.min(1), corner_ys.min(1), corner_xs.max(1), corner_ys.max(1))
            )
            .reshape(4, num_gts)
            .T
        )

        # clip boxes
        new_bboxes[:, 0::2] = new_bboxes[:, 0::2].clip(0, twidth)
        new_bboxes[:, 1::2] = new_bboxes[:, 1::2].clip(0, theight)

        targets[:, :4] = new_bboxes
    else:
        segms = [[trans_points(edge) for edge in obj] for obj in deepcopy(segms)]
        segms = [[np.array([[min(max(x, 0), twidth),
                             min(max(y, 0), theight)]
                            for x, y in edge])
                  for edge in obj]
                 for obj in segms]


        for i, obj in enumerate(segms):
            obj_points = []
            for edge in obj:
                obj_points += edge.tolist()
            obj_points = np.array(obj_points)
            # print(obj_points)
            x_min = obj_points[:, 0].min()
            x_max = obj_points[:, 0].max()
            y_min = obj_points[:, 1].min()
            y_max = obj_points[:, 1].max()
            try:
                targets[i, 0:4] = np.array([x_min, y_min, x_max, y_max])
            except:
                print(i, x_min, y_min, x_max, y_max)
                raise

    return targets, segms


def do_affine(
    img,
    targets=(),
    degrees=10,
    translate=0.1,
    scales=0.1,
    shear=10.,
    segms=None
):

    current_size = img.shape[:2]
    M, scale = get_affine_matrix(current_size, degrees, translate, scales, shear)

    img = cv2.warpAffine(img, M, dsize=current_size, borderValue=(114, 114, 114))

    # Transform label coordinates
    if len(targets) > 0:
        targets, segms = apply_affine_to_bboxes(targets, current_size, M, scale, segms)

    return img, targets, segms, scale


def mirror(image, boxes, prob=0.5, segmentations=None, seg_normed=True):
    height, width, _ = image.shape
    max_wh = max(height, width)
    if random.random() < prob:
        image = np.ascontiguousarray(image[:, ::-1])
        boxes[:, 0:3:2] = width - boxes[:, 2::-2]
        if segmentations is not None:
            segmentations = [[np.array([1. - edge[:, 0] / (1 if seg_normed else max_wh),
                                        edge[:, 1] / (1 if seg_normed else max_wh)]).transpose()
                              for edge in obj]
                             for obj in segmentations]
    else:
        if segmentations is not None and not seg_normed:
            segmentations = [[edge / max_wh for edge in obj] for obj in segmentations]
    return image, boxes, segmentations


def hsv(img, hsv_gain):
    h, s, v = hsv_gain
    if h or s or v:
        r = np.random.uniform(-1, 1, 3) * [h, s, v] + 1  # random gains
        hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
        dtype = img.dtype  # uint8

        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR, dst=img)  # no return needed


# all augmentation methods are gathered here----------------------------------------------------------------------------
class DataAgumentation(Dataset):

    area_threshold = 0.1
    mode = "c"    # a: 1_mosaic+1_normal->mixup   b: 2_mosaic->mixup   c: group_mosaic+1_normal+mixup
    num_branches = 2   # valid only when mode = c.
    current_id = np.array([0])
    current_info = (640, 640)

    def __init__(self,
                 dataset,
                 mosaic=True,
                 degrees=10,
                 translate=0.1,
                 mosaic_scale=(0.5, 1.5),
                 mixup_scale=(0.5, 1.5),
                 shear=2.0,
                 enable_mixup=True,
                 mosaic_prob=1.0,
                 mixup_prob=1.0,
                 rank=0,
                 flip_prob=0.5,
                 hsv_gain=(0.0138, 0.664, 0.464),
                 train_mask=True,
                 *args, **kwargs):

        self.dataset: BaseDataset = dataset

        self.enable_mosaic = mosaic
        self.enable_mixup = enable_mixup
        self.mosaic_prob = mosaic_prob
        self.mixup_prob = mixup_prob
        self.mosaic_scale = mosaic_scale
        self.mixup_scale = mixup_scale

        self.degrees = degrees
        self.translate = translate
        self.shear = shear

        self.flip_prob = flip_prob
        self.hsv_gain = hsv_gain

        self.rank = rank
        self.train_mask = train_mask

        super(DataAgumentation, self).__init__(self.dataset.img_size, mosaic=mosaic)

        img, *_ = self.dataset.pull_item(0)
        self.channel = img.shape[2]
        self.mosaic_border4 = [-self.dataset.img_size[0] // 1, -self.dataset.img_size[1] // 1]
        self.mosaic_border9 = [-self.dataset.img_size[0] // 2, -self.dataset.img_size[1] // 2]

        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def __repr__(self):
        return f"\nData Augment Status:\n" \
               f"Mosaic      : {self.mosaic_prob if self.enable_mosaic else 0}   {self.mosaic_scale}\n" \
               f"Mixup       : {self.mixup_prob if self.enable_mixup else 0}   {self.mixup_scale}\n" \
               f"Rotate      : between ±{self.degrees}°\n" \
               f"Flip        : {'Closed' if self.flip_prob == 0 else self.flip_prob}\n" \
               f"HSV         : {'H: %.3f   S: %.3f   V: %.3f' % self.hsv_gain}\n" \
               f"Translate   : between ±{self.translate}\n" \
               f"Shear       : between ±{self.shear}\n" \
               f"Output Mask : {self.train_mask}\n"

    def __len__(self):
        return len(self.dataset)

    @Dataset.augment_getitem
    def __getitem__(self, index):
        # main branch
        mosaic = self.enable_mosaic and random.random() < self.mosaic_prob
        img, label, segment = self.get_one_branch(index, mosaic, main=True)

        # if use mixup, then get other branches
        if mosaic and self.enable_mixup and random.random() < self.mixup_prob:
            imgs = [img]
            labels = [label]

            img_, label_, segment_ = self.get_one_branch(random.randint(0, len(self)), self.mode == "b", main=False)
            imgs.append(img_)
            labels.append(label_)
            if segment is not None:
                segment.extend(segment_)

            if self.mode == "c":
                for idx in random.choices(range(len(self)), k=self.num_branches-1):
                    img_, label_, segment_ = self.get_one_branch(idx, True, main=False)
                    imgs.append(img_)
                    labels.append(label_)
                    if segment is not None:
                        segment.extend(segment_)
            elif self.mode not in ["a", "b"]:
                raise KeyError

            img, label = self.mixup(imgs, labels)

        # transfer to solid length
        # print(label)

        max_length = self.length_label()
        real_length = len(label)
        labels = -np.ones([max_length, 5])
        labels[:min(real_length, max_length)] = label[:min(real_length, max_length), :5]
        labels = np.ascontiguousarray(labels, dtype=np.float32)
        if segment is None:
            segments = [-1]
        else:
            segments = encode_mask(segment, max_obj_num=max_length, max_point_num=self.dataset.segm_len)

        if self.train_mask:
            return img, labels, self.current_info, self.current_id, segments
        else:
            return img, labels, self.current_info, self.current_id

    def get_one_branch(self, index, mosaic=True, main=False):
        if mosaic:
            img, labels, segments = (self.mosaic4 if random.random() < 0.8 else self.mosaic9)(index, main=main)

            # print("before aug", len(labels), len(segments))
            img, labels, segments = self.augmentation(
                img, labels, segments,
                full_augment=True,
                seg_normed=False
            )
            # print("after aug", len(labels), len(segments))
        else:
            img, labels, segments = self.normal(index, main=main)
            img, labels, segments = self.augmentation(
                img, labels, segments,
                full_augment=False,
                seg_normed=False
            )
        if main:
            return img, labels, segments
        return img, labels, segments

    # 3 types of single branch------------------------------------------------------------------------------------------
    def normal(self, index, main=False):
        img, labels, info, img_id, segments = self.dataset.pull_item(index)

        if main:
            self.current_id = img_id
            self.current_info = info

        sy, sx = self.dataset.img_size
        padded_img = np.full((sy, sx, img.shape[2]), 114, dtype=np.uint8)
        start_x = (sx - img.shape[1]) // 2
        start_y = (sy - img.shape[0]) // 2
        padded_img[start_y:start_y+img.shape[0], start_x:start_x+img.shape[1]] = img

        if len(labels):
            labels = self.update_area(labels, labels)
            labels[:, [0, 2]] += start_x
            labels[:, [1, 3]] += start_y

        if segments is not None:
            sy, sx = self.dataset.img_size
            segments = [
                [np.array([
                    [x * sx + start_x, y * sy + start_y] for x, y in edge
                ]) for edge in obj]
                for obj in segments
            ]

        return padded_img, labels, segments

    def mosaic4(self, index, main=False):
        # loads images in a 4-mosaic

        labels4, segments4 = [], []
        sy, sx = self.dataset.img_size
        max_wh = max(sx, sy)
        yc, xc = [int(random.uniform(-x, 2 * (sx if idx else sy) + x)) for idx, x in enumerate(self.mosaic_border4)]  # mosaic center x, y
        indices = [index] + random.choices(range(len(self)), k=3)  # 3 additional image indices
        img4 = np.full((sy * 2, sx * 2, self.channel), 114, dtype=np.uint8)  # base image with 4 tiles

        for i, index in enumerate(indices):
            # Load image
            # img, _, (h, w) = load_image(self, index)
            img, labels, info, img_id, segments = self.dataset.pull_item(index)
            h, w = img.shape[:2]

            # place img in img4
            if i == 0:  # top left
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
                if main:
                    self.current_id = img_id
                    self.current_info = info
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, sx * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(sy * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, sx * 2), min(sy * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
            else:
                raise ValueError

            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
            padw = x1a - x1b
            padh = y1a - y1b

            # process labels and segments
            if labels.size > 0:
                labels[:, 0] += padw
                labels[:, 1] += padh
                labels[:, 2] += padw
                labels[:, 3] += padh
            labels4.append(labels)

            if segments is not None and segments4 is not None:
                segments = [
                    [np.array([
                        [x * max_wh + padw, y * max_wh + padh] for x, y in edge
                    ]) for edge in obj]
                    for obj in segments
                ]
                segments4.extend(segments)
            else:
                segments4 = None

        out_segment4 = None
        if segments4 is not None:
            out_segment4 = deepcopy(segments4)
            for i, obj in enumerate(out_segment4):
                for j, edge in enumerate(obj):
                    out_segment4[i][j][:, 0] = np.clip(edge[:, 0], 0, 2 * sx)
                    out_segment4[i][j][:, 1] = np.clip(edge[:, 1], 0, 2 * sy)

        if len(labels4):
            labels4 = np.concatenate(labels4, 0)
            out_labels4 = labels4.copy()
            np.clip(out_labels4[:, 0], 0, 2 * sx, out=out_labels4[:, 0])
            np.clip(out_labels4[:, 1], 0, 2 * sy, out=out_labels4[:, 1])
            np.clip(out_labels4[:, 2], 0, 2 * sx, out=out_labels4[:, 2])
            np.clip(out_labels4[:, 3], 0, 2 * sy, out=out_labels4[:, 3])

            if segments4 is not None:
                # print(segments4 == out_segment4, labels4 == out_labels4)
                labels4, segments4 = self.update_area_mask(segments4, out_segment4, labels4, out_labels4)
            else:
                labels4 = self.update_area(labels4, out_labels4)

        return img4, labels4, segments4

    def mosaic9(self, index, main=False):
        labels9, segments9 = [], []
        sy, sx = self.dataset.img_size
        max_wh = max(sx, sy)
        indices = [index] + random.choices(range(len(self)), k=8)  # 8 additional image indices

        hp = wp = h0 = w0 = 0
        img9 = np.full((sy * 3, sx * 3, self.channel), 114, dtype=np.uint8)  # base image with 4 tiles

        for i, index in enumerate(indices):
            # Load image
            img, labels, info, img_id, segments = self.dataset.pull_item(index)
            h, w = img.shape[:2]

            # place img in img9
            if i == 0:  # center
                h0, w0 = h, w
                c = sx, sy, sx + w, sy + h  # xmin, ymin, xmax, ymax (base) coordinates
                if main:
                    self.current_id = img_id
                    self.current_info = info
            elif i == 1:  # top
                c = sx, sy - h, sx + w, sy
            elif i == 2:  # top right
                c = sx + wp, sy - h, sx + wp + w, sy
            elif i == 3:  # right
                c = sx + w0, sy, sx + w0 + w, sy + h
            elif i == 4:  # bottom right
                c = sx + w0, sy + hp, sx + w0 + w, sy + hp + h
            elif i == 5:  # bottom
                c = sx + w0 - w, sy + h0, sx + w0, sy + h0 + h
            elif i == 6:  # bottom left
                c = sx + w0 - wp - w, sy + h0, sx + w0 - wp, sy + h0 + h
            elif i == 7:  # left
                c = sx - w, sy + h0 - h, sx, sy + h0
            elif i == 8:  # top left
                c = sx - w, sy + h0 - hp - h, sx, sy + h0 - hp
            else:
                raise ValueError

            padx, pady = c[:2]
            x1, y1, x2, y2 = [max(x, 0) for x in c]  # allocate coords

            # Labels
            if labels.size > 0:
                labels[:, 0] += padx
                labels[:, 1] += pady
                labels[:, 2] += padx
                labels[:, 3] += pady
            labels9.append(labels)

            if segments is not None and segments9 is not None:
                segments = [
                    [np.array([
                        [x * max_wh + padx, y * max_wh + pady]
                        for x, y in edge
                    ]) for edge in obj]
                    for obj in segments
                ]
                segments9.extend(segments)
            else:
                segments9 = None

            # Image
            img9[y1:y2, x1:x2] = img[y1 - pady:, x1 - padx:]  # img9[ymin:ymax, xmin:xmax]
            hp, wp = h, w  # height, width previous

        # Offset
        yc, xc = [int(random.uniform(0, sy)), int(random.uniform(0, sx))]  # mosaic center x, y
        img9 = img9[yc:yc + 2 * sy, xc:xc + 2 * sx]

        # Concat/clip labels

        c = np.array([xc, yc])  # centers
        out_segment9 = None
        if segments9 is not None:
            segments9 = [[edge - c for edge in obj] for obj in segments9]
            out_segment9 = deepcopy(segments9)
            for i, obj in enumerate(out_segment9):
                for j, edge in enumerate(obj):
                    np.clip(edge[:, 0], 0, 2 * sx, out=out_segment9[i][j][:, 0])
                    np.clip(edge[:, 1], 0, 2 * sy, out=out_segment9[i][j][:, 1])

        if len(labels9):
            labels9 = np.concatenate(labels9, 0)
            labels9[:, [0, 2]] -= xc
            labels9[:, [1, 3]] -= yc
            out_labels9 = labels9.copy()
            np.clip(out_labels9[:, 0], 0, 2 * sx, out=out_labels9[:, 0])
            np.clip(out_labels9[:, 1], 0, 2 * sy, out=out_labels9[:, 1])
            np.clip(out_labels9[:, 2], 0, 2 * sx, out=out_labels9[:, 2])
            np.clip(out_labels9[:, 3], 0, 2 * sy, out=out_labels9[:, 3])

            # count area loss
            if segments9 is not None:
                labels9, segments9 = self.update_area_mask(segments9, out_segment9, labels9, out_labels9)
            else:
                labels9 = self.update_area(labels9, out_labels9)

        return img9, labels9, segments9

    # ------------------------------------------------------------------------------------------------------------------
    # normal augmentation for each type of branch
    def augmentation(self, img, labels, segments, full_augment=True, seg_normed=True):
        if len(labels):
            count = 5
            while True:
                out_img, out_labels, out_segments, scale = do_affine(
                    deepcopy(img),
                    deepcopy(labels),
                    degrees=int(self.degrees / (1 if full_augment else 3)),
                    translate=self.translate / (1 if full_augment else 2.5),
                    scales=self.mosaic_scale if full_augment else self.mixup_scale,
                    shear=self.shear if full_augment else 0,
                    segms=deepcopy(segments)
                )

                current_size = img.shape[:2]
                target_size = self.dataset.img_size

                if current_size == tuple(target_size):
                    x1 = y1 = 0
                    x2, y2 = w, h = target_size
                else:
                    # since translate method is used, we can crop the center of image instead of random crop
                    x1 = int((current_size[1] - target_size[1]) / 2)
                    y1 = int((current_size[0] - target_size[0]) / 2)
                    h, w = target_size
                    x2, y2 = x1 + w, y1 + h

                    out_labels[:, [0, 2]] -= x1
                    out_labels[:, [1, 3]] -= y1

                np.clip(out_labels[:, [0, 2]], 0, w, out=out_labels[:, [0, 2]])
                np.clip(out_labels[:, [1, 3]], 0, h, out=out_labels[:, [1, 3]])
                if segments is not None:
                    for i, obj in enumerate(out_segments):
                        for j, edge in enumerate(obj):
                            edge[:, 0] -= x1
                            edge[:, 1] -= y1
                            np.clip(edge[:, 0], 0, w, out=out_segments[i][j][:, 0])
                            np.clip(edge[:, 1], 0, h, out=out_segments[i][j][:, 1])

                    out_labels, out_segments = self.update_area_mask(segments, out_segments, labels, out_labels, scale)

                else:
                    out_labels = self.update_area(labels, out_labels, scale)

                if len(out_labels) or count == 0:  # 5 chances to get image with labels
                    break
                else:
                    count -= 1

            img = out_img[y1:y2, x1:x2]
            labels = out_labels
            segments = out_segments

            # augmentation without changing labels(except mirror)
            img, labels, segments = mirror(img, labels, self.flip_prob, segments, seg_normed)
            hsv(img, [gain/(1 if full_augment else 4) for gain in self.hsv_gain])

        return img, labels, segments

    # mixup multi branches
    def mixup(self, imgs, labels):
        assert len(imgs) > 1
        if len(imgs) == 2:
            im0, im1 = imgs
            r = np.random.beta(8.0, 8.0)
            imgs = (r * im0.astype(float) + (1 - r) * im1.astype(float)).astype("uint8")
        else:
            n = len(imgs)
            min_ratio = 0.1
            ratios = np.sort(np.random.random(n)) * (1 - min_ratio * n)
            ratios = [ratios[0] + min_ratio] + [ratios[i] - ratios[i-1] + min_ratio for i in range(1, len(ratios))]
            im = imgs.pop(0).astype(float) * ratios.pop(0)

            for im_, r_ in zip(imgs, ratios):
                im += im_.astype(float) * r_
            imgs = im.astype("uint8")

        labels = np.concatenate(labels, 0)
        return imgs, deepcopy(labels)

    # ------------------------------------------------------------------------------------------------------------------
    # count output label length
    def length_label(self):
        mul = 1
        if self.enable_mosaic and self.mosaic_prob > 0:
            mul += 8
        if self.enable_mixup and self.mixup_prob > 0:
            if self.mode == "a":
                mul += 1
            elif self.mode == "b":
                mul += 9
            elif self.mode == "c":
                mul += 9 * self.num_branches + 1
            else:
                raise KeyError
        return mul * self.dataset.max_num_labels

    # update area reserved percentage
    def update_area(self, before, after, scale=1.0):
        ori_areas = count_area(before) * (scale ** 2)
        reserve_areas = count_area(after)

        percentage = reserve_areas / ori_areas

        # when not using segments, affine will make box bigger
        percentage /= max(1, np.max(percentage))

        reserve_mask = percentage > self.area_threshold
        labels = after[reserve_mask]
        if labels.shape[1] == 5:
            labels = np.concatenate([labels, np.array([percentage[reserve_mask]]).transpose()], 1)
        else:
            labels[:, 5] *= percentage[reserve_mask]
        return labels

    # update area reserved percentage via mask
    def update_area_mask(self, seg_before, seg_after, labels_before, labels_after, scale=1.0):
        if seg_after is not None and len(seg_before) == len(labels_before):

            # print(len(seg_before))
            # print(len(seg_after))
            # print(len(labels_before))
            # print(len(labels_after))

            ori_areas = count_area_mask(seg_before) * (scale ** 2)
            valid_mask = ori_areas > 0
            # print(valid_mask)
            ori_areas = ori_areas[np.array(valid_mask)]

            valid_seg_after = []
            for idx, valid in enumerate(valid_mask):
                if valid:
                    valid_seg_after.append(seg_after[idx])
            seg_after = valid_seg_after

            labels_after = labels_after[valid_mask]


            reserve_areas = count_area_mask(seg_after)
            percentage_mask = reserve_areas / ori_areas

            percentage = np.clip(percentage_mask, 0, 1)


            reserve_mask = percentage > self.area_threshold

            out_seg = []
            for flag, obj in zip(reserve_mask, seg_after):
                if flag:
                    out_seg.append(obj)
            labels = labels_after[reserve_mask]
            if labels.shape[1] == 5:
                labels = np.concatenate([labels, np.array([percentage[reserve_mask]]).transpose()], 1)
            else:
                labels[:, 5] *= percentage[reserve_mask]

            return labels, out_seg

        else:
            labels = self.update_area(labels_before, labels_after, scale)
            return labels, None
