import requests
import io
from PIL import Image


def get_image_list(ip, port, **kwargs):
    url = f"http://{ip}:{port}/image_list"
    req = requests.get(url, kwargs)
    # print(req.content)
    # print(req.text)
    # print(req.json())
    return req.json().get("data", [])


def get_image(ip, port, **kwargs):
    url = f"http://{ip}:{port}/image"
    req = requests.get(url, kwargs)
    try:
        data = req.json()
        if "error" not in data:
            data["error"] = "Unknown Error"
        return False, data["error"]
    except Exception as e:
        if "Expecting value" in str(e):
            f = io.BytesIO(req.content)
            image = Image.open(f)
        else:
            raise
    return True, image


def get_categories(ip, port, **kwargs):
    url = f"http://{ip}:{port}/categories"
    req = requests.get(url, kwargs)
    return req.json().get("data", None)


def get_label(ip, port, **kwargs):
    url = f"http://{ip}:{port}/label"
    try:
        req = requests.get(url, kwargs)
        return req.json().get("data", None)
    except:
        return None


def save_label(ip, port, **kwargs):
    url = f"http://{ip}:{port}/save_label"
    req = requests.get(url, kwargs)

    # print(kwargs["label_data"])
    # print(req.text)
    return req.json().get("success", False), req.json().get("error", None)


def test_connect(ip, port, **kwargs):
    url = f"http://{ip}:{port}/test_connect"
    req = requests.get(url, kwargs)
    return req.json().get("data", False), req.json().get("error", None)


if __name__ == '__main__':
    # get_image_list("127.0.0.1", 12345, user="admin", passwd="admin")
    # get_image("127.0.0.1", 12345, user="admin", passwd="admin", name="000000000308.jpg")
    print(get_label("127.0.0.1", 12345, user="admin", passwd="admin", name="000000000308.jpg"))
    # print(save_label("127.0.0.1", 12345, user="admin", passwd="admin", name="000000000308.jpg", label_data=None))
