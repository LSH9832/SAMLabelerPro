import json
from glob import glob
import numpy as np
import datetime
import os
import os.path as osp
import shutil
from threading import Thread
from time import sleep


class COCO:

    def __init__(self, fp=None):
        """
        :param fp: json file
        """
        self.info = {
            'description': "UnNamed",
            'url': "",
            'version': "1.0",
            'year': datetime.datetime.now().year,
            'contributor': "UnNamed",
            'date_created': '%s/%s/%s' % (str(datetime.datetime.now().year),
                                          str(datetime.datetime.now().month).zfill(2),
                                          str(datetime.datetime.now().day).zfill(2))
        }
        self.lic = []
        self.images = []
        self.annotations = []
        self.categories = []
        self.categorie_num = {}
        self.images_dict = {}
        self.real_idx = {}

        if isinstance(fp, str):
            assert os.path.exists(fp)
            fp = open(fp)
        self.load(fp)
        

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        del self.lic
        del self.images
        del self.annotations
        del self.categories
        del self.categorie_num
        return False

    def son_of(self, coco_parent, name="son"):
        """
        Copy messages from parent coco dataset without images and annotations
        :param coco_parent: parent coco dataset
        :param name: name of this dataset, /train/val/test
        """
        self.info = coco_parent.get_info()
        # print(parent.get_info())

        self.info["description"] += "_%s" % name
        self.lic = coco_parent.get_license()
        self.categories = coco_parent.get_categories()
        self.categorie_num = {idx: 0 for idx in coco_parent.get_category_num()}
        
        # print(coco_parent.get_category_num())

    def get_info(self):
        return self.info.copy()

    def get_license(self):
        return self.lic.copy()

    def get_images(self):
        return self.images

    def get_categories(self):
        return self.categories.copy()

    def get_category_num(self):
        return self.categorie_num.copy()

    def _count_category_num(self):
        """
        count and update number of each category
        """
        # print("counting categories")
        self.categorie_num = {}
        for anno in self.annotations:
            if anno['category_id'] - 1 in self.categorie_num:
                self.categorie_num[anno['category_id'] - 1] += 1
            else:
                self.categorie_num[anno['category_id'] - 1] = 1

    def _image_id_exists(self, image_id):
        """
        whether this image id exists
        :param image_id: image id
        """
        flag = False
        for image in self.images:
            if image["id"] == image_id:
                flag = True
                break
        return flag

    def _get_image_data_by_id(self, image_id):
        """
        get image data by its id
        :param image_id:
        :return: image data(dict)
        """
        if image_id in self.images_dict:
            return self.images_dict[image_id]
        for image in self.images:
            if image["id"] == image_id:
                return image
        return None

    def _get_category_id_by_name(self, name, add_mode=True):
        """
        get category id by its name
        :param name: category name
        :return: category id(int)
        """
        for this_category in self.categories:
            if this_category['name'] == name:
                return this_category["id"]
        if add_mode:
            self.add_category(name)
            return self._get_category_id_by_name(name, False)
        else:
            assert add_mode, "category %s not Found" % name

    def _get_name_by_category_id(self, cid):
        for c in self.categories:
            if c["id"] == cid:
                return c["name"]
        return None

    def total_image_number(self):
        return len(self.images)

    def total_annotation_number(self):
        # TODO
        return len(self.annotations)

    def load(self, fp):
        """
        load data from json file
        :param fp: file
        """

        if fp is not None:
            from time import time
            print("loading json...")
            t0 = time()
            data = json.load(fp)
            print("loading time: %.2fs" % (time()-t0))
            self.info = data.get("info", "")
            self.lic = data.get("licenses", "")
            self.images = data["images"]
            self.annotations = data["annotations"]
            self.categories = data["categories"]
            self._count_category_num()
            self._count_real_idx()
            
            for img in self.images:
                self.images_dict[img["id"]] = img
                
    def _count_real_idx(self):
        for i, cate in enumerate(self.categories):
            self.real_idx[cate["id"]] = i
            
    def change_info(self, data_name, version="1.0", url="", author=""):
        """
        change information of this dataset
        :param data_name: name of this dataset
        :param version: version of this dataset
        :param url: url link of this dataset
        :param author: author of this dataset
        """
        self.info = {
            'description': data_name,
            'url': url,
            'version': version,
            'year': datetime.datetime.now().year,
            'contributor': author,
            'date_created': '%s/%s/%s' % (str(datetime.datetime.now().year),
                                          str(datetime.datetime.now().month).zfill(2),
                                          str(datetime.datetime.now().day).zfill(2))
        }

    def add_license(self, name, url=""):
        """
        add license of this dataset
        :param name: name of this license
        :param url: url link of this license
        """
        self.lic.append({
            'url': url,
            'id': len(self.lic) + 1,
            'name': name
        })

    def add_category(self, name, supercategory=None):
        """
        add category of this dataset
        :param name: category name
        :param supercategory: supercategory name
        """
        self.categories.append({
            'supercategory': supercategory,
            'id': len(self.categories) + 1,
            'name': name
        })
        self.categorie_num[len(self.categories) - 1] = 0

    def load_categories(self, txt_file_name):
        """
        load category from txt file
        :param txt_file_name: file name
        """
        if txt_file_name is not None:
            with open(txt_file_name) as fp:
                classes = fp.read().split("\n")
                for this_class in classes:
                    if len(this_class):
                        this_class = this_class.split(":")
                        while this_class[0].endswith(" "):
                            this_class[0] = this_class[0][:-1]
                        self.add_category(name=this_class[0], supercategory=this_class[-1])
        self.categorie_num = {}

    def add_image(
            self,
            image_id,
            file_name,
            width,
            height,
            date_captured=None,
            license_id=1,
            url="",
            flickr_url=None
    ):
        """
        add image data to this dataset
        :param image_id: image id
        :param file_name: file name e.g: 00001.jpg
        :param width: image width
        :param height: image height
        :param date_captured: e.g 2022-02-22 22:22:22
        :param license_id: license id
        :param url: image url
        :param flickr_url: image flickr url
        """
        assert not self._image_id_exists(image_id), "Image ID %d already exists!" % image_id
        self.images.append({
            'license': license_id,
            'file_name': file_name,
            'coco_url': url,
            'height': height,
            'width': width,
            'date_captured': str(datetime.datetime.now()).split(".")[0] if date_captured is None else date_captured,
            'flickr_url': url if flickr_url is None else flickr_url,
            'id': image_id
        })

    def add_annotation(
            self,
            image_id,
            anno_id,
            category_id,
            bbox=None,
            segmentation=None,
            area=None,
            iscrowd=0
    ):
        """
        add annotation of any image exists in this dataset
        :param image_id: image id
        :param anno_id: annotation id
        :param category_id: category id
        :param bbox: bounding box [xmin, ymin, w, h]
        :param segmentation: segmentation [[x00, y00, x01, y01, ....], [x10, y10, x11, y11, ....], ....]
        :param area: area of segmentation if segmentation is not empty else area of bounding box
        :param iscrowd: is crowd
        """
        assert bbox or segmentation, "bbox or segmentation is required"
        assert self._image_id_exists(image_id), "Image ID %d does not exist!" % image_id

        if bbox is None:
            bbox = []
        if segmentation is None:
            segmentation = []
        if area is None and len(bbox):
            area = bbox[2] * bbox[3]

        self.annotations.append({
            'segmentation': segmentation,
            'area': area,
            'iscrowd': iscrowd,
            'image_id': image_id,
            'bbox': bbox,
            'category_id': category_id,
            'id': anno_id
        })
        if category_id - 1 in self.categorie_num:
            self.categorie_num[category_id - 1] += 1
        else:
            self.categorie_num[category_id - 1] = 1

    def save(self, file_name: str = None):
        """
        save data to a json file
        :param file_name: file name with path
        """
        file_name = self.info["description"] if file_name is None else file_name
        file_name = file_name if file_name.endswith(".json") else "%s.json" % file_name
        json.dump({
            'info': self.info,
            'licenses': self.lic,
            'images': self.images,
            'annotations': self.annotations,
            'categories': self.categories
        }, open(file_name, "w"))
        print("coco annotation saved to %s." % os.path.abspath(file_name).replace("\\", "/"))

    def show_each_category_num(self, width=50, simple=False):
        """
        show number of each category in a table
        :param width: tabel width
        :param simple: show simple table
        """
        if simple:
            categories_str = ""
            for cate in self.categories:
                categories_str += cate["name"].ljust(width//3 * 2) + \
                                  str(self.categorie_num[cate["id"] - 1]
                                      if (cate["id"] - 1) in self.categorie_num else 0).rjust(width - width//3*2) + "\n"
            return f"""
{"=" * width}
Category Count
{"%s images" % str(len(self.images)).rjust(10)}
{"%s annotations" % str(len(self.annotations)).rjust(10)}
{"=" * width}
{categories_str}{"=" * width}"""
        width = max(28, int(width))
        head = "╒%s╕\n" \
               "│%sCategory Count%s│\n" \
               "╞%s╡\n" % ("═" * width, " " * int((width - 14) / 2), " " * int((width - 14) / 2), "═" * width)
        msg = ""
        msgs = ["%s images" % str(len(self.images)).rjust(10),
                "%s annotations" % str(len(self.annotations)).rjust(10)]
        for this_msg in msgs:
            msg += "│%s│\n" % this_msg.ljust(width)

        neck = "╞%s╤%s╡\n" % ("═" * (width - 1 - int(width / 3)), "═" * int(width / 3))



        body = ""
        for cate in self.categories:
            body += "│%s│%s│\n" % (
                cate["name"].ljust(width - 1 - int(width / 3)),
                str(self.categorie_num[cate["id"] - 1] if (cate["id"] - 1) in self.categorie_num else 0).rjust(int(width / 3))
            )
            # print("│%s│%s│" % (
            #     cate["name"].ljust(width - 1 - int(width / 3)),
            #     str(self.categorie_num[cate["id"] - 1] if (cate["id"] - 1) in self.categorie_num else 0).rjust(int(width / 3))
            # ))

            if self.categories.index(cate) < len(self.categories) - 1:
                body += "├%s┼%s┤\n" % ("─" * (width - 1 - int(width / 3)), "─" * int(width / 3))
                # print("├%s┼%s┤" % ("─" * (width - 1 - int(width / 3)), "─" * int(width / 3)))
            else:
                body += "╘%s╧%s╛" % ("═" * (width - 1 - int(width / 3)), "═" * int(width / 3))
                # print("╘%s╧%s╛" % ("═" * (width - 1 - int(width / 3)), "═" * int(width / 3)))

        show_str = head + msg + neck + body
        print()
        print(show_str)
        print()

        return show_str


def to_coco_segments(edge: list):
    ret = []
    for x, y in edge:
        ret.extend([int(round(x)), int(round(y))])
    return ret


def gather_box(box1, box2):
    boxes = np.round(np.array([box1, box2])).astype(int)
    return [int(np.min(boxes[:, 0])), int(np.min(boxes[:, 1])), int(np.max(boxes[:, 2])), int(np.max(boxes[:, 3]))]


def decode_json(file):
    data = json.load(open(file))
    ret = {
        "img_name": data["info"]["name"],
        "img_info": {
            "width": data["info"]["width"],
            "height": data["info"]["height"]
        },
        "objs": []
    }

    objs = {}
    for obj in data["objects"]:

        if obj["category"] not in objs:
            objs[obj["category"]] = {}
            objs[obj["category"]][None] = []

        if isinstance(obj["group"], int) or obj["group"].isdigit():
            obj["group"] = int(obj["group"])
            if obj["group"] not in objs[obj["category"]]:
                objs[obj["category"]][obj["group"]] = {
                    "bbox": [int(loc) for loc in obj["bbox"]],
                    "segmentation": [to_coco_segments(obj["segmentation"])],
                    "area": obj["area"],
                    "iscrowd": obj["iscrowd"]
                }
            else:
                objs[obj["category"]][obj["group"]]["area"] += obj["area"]
                objs[obj["category"]][obj["group"]]["iscrowd"] = obj["iscrowd"] or objs[obj["category"]][obj["group"]]["iscrowd"]
                objs[obj["category"]][obj["group"]]["segmentation"].append(to_coco_segments(obj["segmentation"]))
                objs[obj["category"]][obj["group"]]["bbox"] = gather_box(obj["bbox"], objs[obj["category"]][obj["group"]]["bbox"])
        else:
            objs[obj["category"]][None].append(
                {
                    "bbox": [int(loc) for loc in obj["bbox"]],
                    "segmentation": [to_coco_segments(obj["segmentation"])],
                    "area": obj["area"],
                    "iscrowd": obj["iscrowd"]
                }
            )

    for name, v in objs.items():
        for group, data in v.items():
            if group is None:
                for obj in data:
                    obj["name"] = name
                    bbox = obj["bbox"]
                    obj["bbox"] = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]
                    ret["objs"].append(obj)
            else:
                data["name"] = name
                bbox = data["bbox"]
                data["bbox"] = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]
                ret["objs"].append(data)

    return ret


def main():
    to_coco(
        image_dirs="example/images",
        annotation_dirs="example/images",
        data_name="example",
        image_target_dir="example/images/coco/example/images",
        dist_dir="example/images/coco/example"
    )


def to_coco(image_dirs,
            annotation_dirs,
            class_file=None,
            data_name=None,
            image_target_dir: str = None,
            image_type="jpg",
            version="1.0",
            url="",
            author="",
            save=True,
            dist_dir="./",
            coco_dataset: COCO = None,
            max_num=1000,
            q=None) -> COCO:

    if coco_dataset is None:
        coco_dataset = COCO()
        coco_dataset.change_info(data_name=data_name, version=version, url=url, author=author)
        coco_dataset.add_license(name="FAKE LICENSE", url="")
        coco_dataset.load_categories(txt_file_name=class_file)

    os.makedirs(dist_dir, exist_ok=True)
    os.makedirs(osp.join(dist_dir, "annotations"), exist_ok=True)
    os.makedirs(image_target_dir, exist_ok=True)


    labels_list = glob(osp.join(annotation_dirs, "*.json").replace("\\", "/"))

    start = coco_dataset.total_image_number()
    anno_id = coco_dataset.total_annotation_number()
    # start = image_id
    minous = 0
    anno_file_num = len(labels_list)
    # max_num = 1000

    q.put(anno_file_num)
    
    


    def one_thread(index, total):
        nonlocal anno_id, minous, coco_dataset

        count = 0

        for i, label_name in enumerate(labels_list):

            if not i % total == index:
                continue

            data = decode_json(label_name)


            img_file = osp.join(image_dirs, data["img_name"]).replace("\\", "/")

            if q is not None:
                count += 1
                if count == 10:
                    q.put(count)
                    count = 0

            if osp.isfile(img_file):

                image_id = start + i + 1 - minous
                image_name = str(image_id).zfill(10) + ".%s" % image_type
                shutil.copyfile(img_file, os.path.join(image_target_dir, image_name))

                print("\rConverting: %d / %d" % (image_id, anno_file_num + start), end="")

                coco_dataset.add_image(
                    image_id=image_id,
                    file_name=image_name,
                    **data["img_info"]
                )

                for obj in data["objs"]:
                    obj: dict
                    name = obj.pop("name")
                    anno_id += 1

                    coco_dataset.add_annotation(
                        image_id=image_id,
                        anno_id=anno_id,
                        category_id=coco_dataset._get_category_id_by_name(name),
                        **obj
                    )
            else:
                minous += 1
                print(f"image {img_file} does not exist!")

        if q is not None:
            q.put(count)


    num_threads = min(32, os.cpu_count())
    threads = [Thread(target=one_thread, args=(i, num_threads)) for i in range(num_threads)]
    [thread.start() for thread in threads]
    [thread.join() for thread in threads]

    print("\nConvert finished. Images saved to %s" % image_target_dir.replace("\\", "/"))
    coco_dataset.save(osp.join(dist_dir, "annotations", f"{data_name}_{osp.dirname(image_target_dir)}")) if save else None

    if q is not None:
        try:
            dir_name = osp.join(dist_dir, "annotations")
            json_log = os.path.join(dir_name, "category_msg.txt")
            with open(json_log, "w") as f:
                f.write(coco_dataset.show_each_category_num())
            label_file = os.path.join(dir_name, "classes.txt")
            label_str = ""
            for label in coco_dataset.get_categories():
                label_str += f"{label['name']}\n"
            with open(label_file, "w") as f:
                f.write(label_str[:-1])
            q.put(coco_dataset.show_each_category_num(35, True))
        except:
            pass

    return coco_dataset


def divide_coco_by_image(json_file, train: float, val: float, q=None):
    assert 0. < train < 1. and 0. < val < 1. and 0. < train + val <= 1.

    import random

    try:

        # print(json_file)

        coco_dataset = COCO(json_file)
        datasets = {
            "train": COCO(),
            "val": COCO(),
            "test": COCO()
        }
        for item in datasets:
            datasets[item].son_of(coco_dataset, name=item)

        image_map = [i for i in range(len(coco_dataset.images))]
        random.shuffle(image_map)
        train_img_num = round(len(coco_dataset.images) * train)
        val_img_num = round(len(coco_dataset.images) * val)

        for anno in coco_dataset.annotations:
            # anno['image_id']
            img_id = anno['image_id'] - 1
            this_rank = image_map.index(img_id)

            kw = "test"
            if this_rank < train_img_num:
                kw = "train"
            elif this_rank < train_img_num + val_img_num:
                kw = "val"

            datasets[kw].annotations.append(anno)
            if not datasets[kw]._image_id_exists(anno['image_id']):
                datasets[kw].images.append(coco_dataset._get_image_data_by_id(anno['image_id']))

        for item in datasets:
            datasets[item]._count_category_num()
            print(item)
            print(datasets[item].show_each_category_num(35, True))
            datasets[item].save(file_name=osp.join(osp.dirname(json_file), datasets[item].info["description"]))

        if q is not None:
            ret_str = ""
            for k, v in datasets.items():
                ret_str += v.show_each_category_num(35, True) + "\n"
            q.put(ret_str)

        return datasets
    except Exception as e:
        print(e)


def coco2yolo(json_file, dist_dir, moveImg=False, oriImgPath="", distImgPath="", q=None, num_process=1, rank=0):
    global count
    coco_dataset = COCO(json_file)
    
    yolo_labels = {}
    
    thread_num = min(32, os.cpu_count())
    num_anno = len(coco_dataset.annotations)
    
    count = 0
    dist_dir = osp.abspath(dist_dir).replace('\\', '/')
    if not osp.isdir(dist_dir):
        os.makedirs(dist_dir, exist_ok=True)
    
    if moveImg:
        os.makedirs(distImgPath, exist_ok=True)
        (print if q is None else q.put)(f"start moving images to {distImgPath}")
    
    num_files = len(coco_dataset.images)
    bar_length = num_files + num_anno * (2 if moveImg else 1)
    output_files = []
    count = 0
    print()
    for img in coco_dataset.images:
        img_suffix = img["file_name"].split(".")[-1]
        output_files.append(osp.join(dist_dir, osp.basename(img["file_name"])[:-len(img_suffix)-1] + ".txt").replace("\\", "/"))

        img_file = osp.basename(img["file_name"])
        ori_img_path = osp.join(oriImgPath, img_file)
        
        if moveImg:
            count += 1
            print(f"\rmove files: {count}/{num_files}      ", end="")
            if osp.isfile(ori_img_path):
                shutil.move(ori_img_path, distImgPath)
                if q is not None:
                    q.put([rank, count / bar_length, bar_length])

    print()
    count = 0
    
    def label_deal_thread(thread_rank):
        global count
        for i, anno in enumerate(coco_dataset.annotations):
            if i % thread_num == thread_rank:
                img_info = coco_dataset._get_image_data_by_id(anno['image_id'])
                img_suffix = img_info["file_name"].split(".")[-1]
                label_fname = osp.join(dist_dir, osp.basename(img_info["file_name"])[:-len(img_suffix)-1] + ".txt").replace("\\", "/")
                
                label = coco_dataset.real_idx[anno["category_id"]]
                bbox = anno["bbox"]
                img_w = img_info["width"]
                img_h = img_info["height"]
                
                cx, cy, w, h = [bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2, bbox[2], bbox[3]]
                cx /= img_w
                cy /= img_h
                w /= img_w
                h /= img_h
                
                if label_fname in yolo_labels:
                    yolo_labels[label_fname] += f"\n{label} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
                else:
                    yolo_labels[label_fname] = f"{label} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
                count += 1
                print(f"\rconverting labels: {count}/{num_anno}           ", end="")
                if q is not None:
                    if count == num_anno or count % 100 == 0:
                        q.put([rank, (count + (num_files if moveImg else 0)) / bar_length, bar_length])
    print()
    (print if q is None else q.put)(f"start dealing labels, number in total: {num_anno}")
    
    threads = [Thread(target=label_deal_thread, args=(k, )) for k in range(thread_num)]
    [thread.start() for thread in threads]
    [thread.join() for thread in threads]
    
    print()
    
    (print if q is None else q.put)(f"start writing labels files, image number with at least 1 label in total: {len(yolo_labels.keys())}/{len(output_files)}")
    
    
    count = 0
    
    empty_files = ""
    for this_file in output_files:
        content = yolo_labels.get(this_file, "")
        if not len(content):
            empty_files += f"\n{osp.basename(this_file)} is empty"
        
        with open(this_file, "w") as label_file:
            label_file.write(content)
        count += 1
        print(f"\rwriting files: {count}/{num_files}    ", end="")
        if q is not None:
            if count == num_files or count % 100 == 0:
                q.put([rank, (count + num_anno + (num_files if moveImg else 0)) / bar_length, bar_length])

    print()
    # for k, v in yolo_labels.items():
    #     open(k, "w").write(v)
    
    (print if q is None else q.put)(empty_files)
    (print if q is None else q.put)(f"all labels are saved to {dist_dir}")


if __name__ == '__main__':
    # main()
    
    coco2yolo(
        "E:/dataset/coco2017/annotations/instances_train2017.json",
        "E:/dataset/coco2017/labels/train"
    )
