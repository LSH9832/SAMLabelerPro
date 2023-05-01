from .datasets_wrapper import Dataset
import cv2
import numpy as np


class BaseDataset(Dataset):

    max_num_labels = 80
    segm_len = 0

    def __init__(self, input_size, mosaic=False):
        super(BaseDataset, self).__init__(input_size, mosaic=mosaic)
        self.img_size = input_size
        self.annotation_list = []

    def __len__(self):
        return len(self.annotation_list)

    def __del__(self):
        try:
            del self.annotation_list
        except Exception as e:
            print(e)
            self.annotation_list = []

    def load_anno(self, index):
        return self.annotation_list[index]["annotations"]   # [num_obj, 5(xywh + cls)]

    def load_resized_img(self, index, res):
        img = self.load_image(index)
        r = min(self.img_size[0] / img.shape[0], self.img_size[1] / img.shape[1])

        img_h, img_w = img.shape[:2]

        # print(res)

        res[..., 0] *= img_w
        res[..., 2] *= img_w
        res[..., 1] *= img_h
        res[..., 3] *= img_h

        res[..., :4] *= r

        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)

        return resized_img, res, (img.shape[0], img.shape[1])

    def load_image(self, index):
        img_file = self.annotation_list[index]["image"].replace('\\', '/')
        img = cv2.imread(img_file)
        assert img is not None, f"File {img_file} does not exist or broken!"
        return img

    def pull_item(self, index):
        """
        Returns:
          resized_img, rectangles, origin_img_size, idx, segments
        """
        anno = self.annotation_list[index]
        res = anno["annotations"].copy()

        img, res, img_info = self.load_resized_img(index, res)

        return img, res, img_info, np.array([index]), None

    def __getitem__(self, item):
        raise NotImplementedError
