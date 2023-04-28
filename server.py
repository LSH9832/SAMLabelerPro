from flask import Flask, request, send_file
from glob import glob
import os
import os.path as osp
import yaml
import json
import datetime
import argparse


def make_parser():
    parser = argparse.ArgumentParser("SAMLabeler Pro Image Server Parser")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="host ip")
    parser.add_argument("-p", "--port", type=int, default=12345, help="server port")
    parser.add_argument("--debug", action="store_true", help="debug mode")
    return parser.parse_args()


app = Flask(__name__)

suffixs = ('bmp', 'cur', 'gif', 'icns', 'ico', 'jpeg', 'jpg', 'pbm', 'pgm', 'png', 'ppm', 'svg',
           'svgz', 'tga', 'tif', 'tiff', 'wbmp', 'webp', 'xbm', 'xpm')


def get_server_data():
    return yaml.load(open("settings/server_settings.yaml"), yaml.SafeLoader)


def check_(user, pwd):
    users_list = get_server_data()["users"]
    return user in users_list and users_list[user]["pwd"] == str(pwd) and pwd is not None


def get_args():
    return request.args.to_dict()


@app.route("/image_list", methods=["GET", "POST"])
def image_list():

    args = get_args()

    if not check_(args.get("user", None), args.get("passwd", None)):
        return {"data": [], "error": "user or passwd is wrong"}

    server_data = get_server_data()
    image_path = server_data["users"][args.get("user")]["image_path"]
    average = server_data["average"]
    num_workers = 1
    user_range = 1
    if average:
        num_workers = 0
        for user, v in server_data["users"].items():

            if v["image_path"] == image_path:
                num_workers += 1
                if user == args.get("user"):
                    user_range = num_workers - 1

    files = []
    for suffix in suffixs:
        files.extend(glob(osp.join(image_path, f"*.{suffix}").replace("\\", "/")))

    if average:
        worker_files = []
        for i, f in enumerate(sorted(files)):
            if i % num_workers == user_range:
                worker_files.append(osp.basename(f))
    else:
        worker_files = [osp.basename(f) for f in sorted(files)]
    return {"data": worker_files}


@app.route("/image", methods=["GET", "POST"])
def image():

    args = get_args()

    if not check_(args.get("user", None), args.get("passwd", None)):
        return {"data": None, "error": "user or passwd is wrong"}

    server_data = get_server_data()
    image_path = server_data["users"][args.get("user")]["image_path"]

    file = osp.join(image_path, args.get("name", "no_implement"))
    if osp.isfile(file):
        support = False
        for suffix in suffixs:
            if file.endswith(suffix):
                support = True
                break
        if support:
            return send_file(file)
        else:
            return {"data": None, "error": "type not support"}
    else:
        return {"data": None, "error": "no such file"}


@app.route("/categories", methods=["GET", "POST"])
def categories():
    args = get_args()

    server_data = get_server_data()
    category_file = server_data["users"][args.get("user")]["category_file"]

    if not check_(args.get("user", None), args.get("passwd", None)):
        return {"data": None, "error": "user or passwd is wrong"}

    return {"data": yaml.load(open(category_file), yaml.SafeLoader)}


@app.route("/label", methods=["GET", "POST"])
def label():
    args = get_args()

    if not check_(args.get("user", None), args.get("passwd", None)):
        return {"data": None, "error": "user or passwd is wrong"}

    server_data = get_server_data()
    label_path = server_data["users"][args.get("user")]["label_path"]

    name = args.get("name")
    if name is not None and "." in name:
        json_file = name.split(".")[0] + ".json"
        json_path = osp.join(label_path, json_file)
        data = None
        if osp.isfile(json_path):
            data = json.load(open(json_path))
        return {"data": data}
    else:
        return {"data": None, "error": "name is wrong"}


@app.route("/test_connect", methods=["GET", "POST"])
def test_connect():
    args = get_args()
    if not check_(args.get("user", None), args.get("passwd", None)):
        return {"data": False, "error": "user or passwd is wrong"}
    return {"data": True}


@app.route("/save_label", methods=["GET", "POST"])
def save_label():
    args = get_args()

    if not check_(args.get("user", None), args.get("passwd", None)):
        return {"success": False, "error": "user or passwd is wrong"}
    try:
        server_data = get_server_data()
        image_path = server_data["users"][args.get("user")]["image_path"]
        label_path = server_data["users"][args.get("user")]["label_path"]
        os.makedirs(label_path, exist_ok=True)

        name = args.get("name")
        image_file = osp.join(image_path, name)
        if not osp.isfile(image_file):
            return {"success": False, "error": "relative image not exist!"}
        if name is not None and "." in name:
            json_file = name.split(".")[0] + ".json"
            json_path = osp.join(label_path, json_file)
            data = args.get("label_data", None)
            if data is not None:
                data = json.loads(data)
                data["info"]["note"] = f'(Edit by user "{args.get("user", None)}" at ' \
                                       f'{datetime.datetime.now().strftime("%Y-%m-%D %H:%M:%S, %A %B")})'
                json.dump(data, open(json_path, "w"))
                return {"success": True}
            else:
                return {"success": False, "error": "server did not receive label data"}
        else:
            return {"success": False, "error": "name is wrong"}
    except Exception as e:
        return {"success": False, "error": str(e)}


if __name__ == '__main__':
    args = make_parser()
    app.run(args.host, args.port, args.debug)
