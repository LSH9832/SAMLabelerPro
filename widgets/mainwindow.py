import time

import numpy as np
import yaml
import json
from glob import glob
from PyQt5 import QtWidgets, QtCore, QtGui
from ui.MainWindow import Ui_MainWindow
from widgets.setting_dialog import SettingDialog
from widgets.category_choice_dialog import CategoryChoiceDialog
from widgets.category_edit_dialog import CategoryEditDialog
from widgets.labels_dock_widget import LabelsDockWidget
from widgets.files_dock_widget import FilesDockWidget
from widgets.info_dock_widget import InfoDockWidget
from widgets.right_button_menu import RightButtonMenu
from widgets.shortcut_dialog import ShortcutDialog
from widgets.about_dialog import AboutDialog
from widgets.converter import ConvertDialog
from widgets.canvas import AnnotationScene, AnnotationView
from widgets.remote_settings import RemoteSettings
from configs import (
    STATUSMode,
    MAPMode,
    load_config,
    save_config,
    CONFIG_FILE,
    DEFAULT_CONFIG_FILE,
    DEFAULT_TITLE,
    REMOTE_CONFIG_FILE,
    DEFAULT_EDIT_CONFIG
)
from annotation import Object, Annotation
from widgets.polygon import Polygon
import os
import os.path as osp
from PIL import Image
import functools
import imgviz
from segment_any.segment_any import SegAny
from segment_any.gpu_resource import GPUResource_Thread, osplatform
from label2coco import LabelConverter
import icons_rc

import utils.remote as remote


class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):

    single_vis = False
    selected_item = None

    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)

        ######################################################
        self.setWindowIcon(self.win_icon)
        self._edit_setting_file = DEFAULT_EDIT_CONFIG   # "settings/last_edit.yaml"
        self.edit_data = {
            "image_dir": None,
            "label_dir": None,
            "current_index": 0,
            "half": True,
            "cfg": "settings/coco.yaml" if osp.isfile("settings/coco.yaml") else CONFIG_FILE if osp.exists(CONFIG_FILE) else DEFAULT_CONFIG_FILE,
            "force_model_type": None,
            "first": True,
            "remote": False,
            "remote_data": {
                "ip": "127.0.0.1",
                "port": 12345,
                "user": "admin",
                "passwd": "admin",
                "remote_index": 0
            }
        }
        if os.path.isfile(self._edit_setting_file):
            for k, v in yaml.load(open(self._edit_setting_file), yaml.SafeLoader).items():
                self.edit_data[k] = v
        else:
            os.makedirs("settings", exist_ok=True)
        self.save_current_state()
        ######################################################

        self.init_ui()
        self.image_root: str = None
        self.label_root: str = None

        self.files_list: list = []
        self.current_index = None
        self.current_file_index: int = None

        self.local_cfg = self.edit_data.get("cfg")
        self.config_file = self.edit_data.get("cfg") if not self.edit_data["remote"] else REMOTE_CONFIG_FILE
        self.saved = True
        self.load_finished = False
        self.polygons: list = []

        self.png_palette = None     # 图像拥有调色盘，说明是单通道的标注png文件
        self.instance_cmap = imgviz.label_colormap()
        self.map_mode = MAPMode.LABEL
        # 标注目标
        self.current_label: Annotation = None

        self.reload_cfg()

        self.init_connect()
        self.reset_action()

        self.setMinimumSize(1600, 900)
        self.view.setMinimumSize(1280, 720)
        # print(self.cfg)
        self.setWindowTitle(DEFAULT_TITLE + self.title_with_remote())
        self.init_segment_anything()
        self.reload_mode(False)

        self.heart_beat_timer = QtCore.QTimer(self)
        self.heart_beat_timer.timeout.connect(self.__heart_beat)
        self.heart_beat_timer.start(2000)

    def reload_mode(self, reload_config=True):
        self.config_file = self.edit_data.get("cfg") if not self.edit_data["remote"] else REMOTE_CONFIG_FILE

        if self.edit_data.get("image_dir") is not None or self.edit_data["remote"]:
            self.open_dir(dir=self.edit_data.get("image_dir"), show=False)
            if not self.edit_data["remote"]:
                self.save_dir(dir=self.edit_data.get("label_dir"), show=False)

        self.setVisible(True)
        if len(self.files_list) and self.current_index is not None:
            self.show_image(self.current_index)

        if self.edit_data.get("first", True):
            self.shortcut_dialog.show()
            self.edit_data["first"] = False
            self.save_current_state()
        if reload_config:
            self.reload_cfg()
        self.actionOpen_dir.setEnabled(not self.edit_data["remote"])
        self.actionSave_dir.setEnabled(not self.edit_data["remote"])

    def title_with_remote(self):
        self.cfg["language"] = self.cfg.get("language", "en")
        return (f"(Remote Mode: {self.edit_data.get('remote_data').get('ip')}:"
                f"{self.edit_data.get('remote_data').get('port')})"
                if self.cfg["language"] == "en" else
                f"(远程模式：{self.edit_data.get('remote_data').get('ip')}:"
                f"{self.edit_data.get('remote_data').get('port')})") \
            if self.edit_data["remote"] else \
            ("(Local Mode)" if self.cfg.get("language", "en") == "en" else "(本地模式)")

    def save_current_state(self):
        yaml.dump(self.edit_data, open(self._edit_setting_file, "w"), yaml.Dumper)

    def init_segment_anything(self):

        print("init SAM")

        weights_list = glob("**/*.pth", recursive=True)
        mobile_weights_list = glob("**/*.pt", recursive=True)
        self.use_segment_anything = False

        # print(weights_list)

        if len(weights_list+mobile_weights_list):
            weights_size = [99999 for _ in mobile_weights_list] + \
                           [os.path.getsize(file) / 1024 ** 3 for file in weights_list]

            for _, file in sorted(zip(weights_size, mobile_weights_list + weights_list), reverse=True):
                try:
                    file = file.replace("\\", "/")
                    self.segany = SegAny(file, self.edit_data.get("half", True), self.edit_data.get("force_model_type"))
                    if self.segany.success:
                        self.statusbar.showMessage(f'Using weights: {file}.')
                        self.use_segment_anything = True
                        break
                except Exception as e:
                    # print(e)
                    pass

        if not self.use_segment_anything:
            websites = 'https://github.com/facebookresearch/segment-anything#model-checkpoints'
            try:
                self.heart_beat_timer.stop()
            except:
                pass
            QtWidgets.QMessageBox.warning(
                self, 'Warning',
                f'The checkpoint of [Segment anything] not existed. If you want use quick annotate, '
                f'please download from {websites}'
            )
            try:
                self.heart_beat_timer.start(2000)
            except:
                pass

        if self.use_segment_anything:
            if self.segany.device != 'cpu':
                self.gpu_resource_thread = GPUResource_Thread()
                self.gpu_resource_thread.message.connect(self.labelGPUResource.setText)
                self.gpu_resource_thread.start()
            else:
                self.labelGPUResource.setText('cpu')
        else:
            self.labelGPUResource.setText('segment anything unused.')

    def init_ui(self):
        self.setting_dialog = SettingDialog(parent=self, mainwindow=self)
        self.remote_settings_win = RemoteSettings(parent=self)

        self.labels_dock_widget = LabelsDockWidget(mainwindow=self)
        self.labels_dock.setWidget(self.labels_dock_widget)

        self.files_dock_widget = FilesDockWidget(mainwindow=self)
        self.files_dock.setWidget(self.files_dock_widget)

        self.info_dock_widget = InfoDockWidget(mainwindow=self)
        self.info_dock.setWidget(self.info_dock_widget)

        self.scene = AnnotationScene(mainwindow=self)
        self.category_choice_widget = CategoryChoiceDialog(self, mainwindow=self, scene=self.scene)
        self.category_edit_widget = CategoryEditDialog(self, self, self.scene)

        self.convert_dialog = ConvertDialog(self, mainwindow=self)
        self.convert_coco_win = LabelConverter(parent=self)

        self.view = AnnotationView(parent=self)
        self.view.setScene(self.scene)
        self.setCentralWidget(self.view)

        self.right_button_menu = RightButtonMenu(mainwindow=self)
        self.right_button_menu.addAction(self.actionEdit)
        self.right_button_menu.addAction(self.actionTo_top)
        self.right_button_menu.addAction(self.actionTo_bottom)

        self.shortcut_dialog = ShortcutDialog(self)
        self.about_dialog = AboutDialog(self)

        self.labelGPUResource = QtWidgets.QLabel('')
        self.labelGPUResource.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        self.labelGPUResource.setFixedWidth(280)
        self.statusbar.addPermanentWidget(self.labelGPUResource)

        self.labelCoord = QtWidgets.QLabel('')
        self.labelCoord.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        self.labelCoord.setFixedWidth(150)
        self.statusbar.addPermanentWidget(self.labelCoord)

        self.labelData = QtWidgets.QLabel('')
        self.labelData.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        self.labelData.setFixedWidth(150)
        self.statusbar.addPermanentWidget(self.labelData)

        self.trans = QtCore.QTranslator()

    def translate(self, language='zh'):
        if language == 'zh':
            self.trans.load('ui/zh_CN')
        else:
            self.trans.load('ui/en')
        self.actionChinese.setChecked(language=='zh')
        self.actionEnglish.setChecked(language=='en')
        _app = QtWidgets.QApplication.instance()
        _app.installTranslator(self.trans)
        self.retranslateUi(self)

        self.info_dock_widget.retranslateUi(self.info_dock_widget)
        self.labels_dock_widget.retranslateUi(self.labels_dock_widget)
        self.files_dock_widget.retranslateUi(self.files_dock_widget)
        self.category_choice_widget.retranslateUi(self.category_choice_widget)
        self.category_edit_widget.retranslateUi(self.category_edit_widget)
        self.setting_dialog.retranslateUi(self.setting_dialog)
        self.about_dialog.retranslateUi(self.about_dialog)
        self.shortcut_dialog.retranslateUi(self.shortcut_dialog)
        self.convert_dialog.retranslateUi(self.convert_dialog)

        self.actionSetting.setText("Label Setting" if language == "en" else "标签设置")
        self.actionSetting.setStatusTip("Label Setting." if language == "en" else "标签设置。")
        self.actionRemoteSetting.setText("Remote Setting" if language == "en" else "远程模式设置")
        self.actionRemoteSetting.setStatusTip("Remote Setting." if language == "en" else "远程模式设置。")

    def translate_to_chinese(self):
        self.translate('zh')
        self.cfg['language'] = 'zh'

    def translate_to_english(self):
        self.translate('en')
        self.cfg['language'] = 'en'

    def reload_cfg(self):

        if not self.edit_data["remote"]:
            self.edit_data["cfg"] = self.config_file
        self.cfg = load_config(self.config_file if osp.isfile(self.config_file) else DEFAULT_CONFIG_FILE)
        self.cfg["label"] = self.cfg.get('label', [])

        if self.edit_data["remote"]:
            import requests
            try:
                remote_cfg = remote.get_categories(**self.edit_data["remote_data"])
            except requests.exceptions.ConnectionError:
                # print(e)
                remote_cfg = None
                self.edit_data["remote"] = False
                self.edit_data["cfg"] = self.config_file = self.local_cfg
                self.cfg = load_config(self.config_file if osp.isfile(self.config_file) else DEFAULT_CONFIG_FILE)
                self.cfg["label"] = self.cfg.get('label', [])
                print("can not connect to the remote server, change to local mode.")

            if remote_cfg is not None:
                exist_names = [cate["name"] for cate in self.cfg["label"]]
                for category in remote_cfg["label"]:
                    if category["name"] in exist_names:
                        self.cfg["label"][exist_names.index(category["name"])] = category
                    else:
                        self.cfg["label"].append(category)

        language = self.cfg.get('language', 'en')
        self.translate(language)

        label_dict_list = self.cfg.get('label', [])
        d = {}
        for label_dict in label_dict_list:
            category = label_dict.get('name', 'unknow')
            color = label_dict.get('color', '#000000')
            d[category] = color
        self.category_color_dict = d

        if self.current_index is not None:
            self.show_image(self.current_index)

    def set_saved_state(self, is_saved:bool):
        self.saved = is_saved
        if self.files_list is not None and self.current_index is not None:

            if is_saved:
                self.setWindowTitle(f'{DEFAULT_TITLE+self.title_with_remote()} - {self.current_label.label_path}')
            else:
                self.setWindowTitle(f'{DEFAULT_TITLE+self.title_with_remote()} - *{self.current_label.label_path}')

    def open_dir(self, dir=None, show=True):
        reset_index = dir is None and not self.edit_data["remote"]
        if self.edit_data["remote"]:
            remote_data = self.edit_data["remote_data"]
            self.files_list = remote.get_image_list(**remote_data)
            self.files_dock_widget.update_widget()
            print("远程获取完毕")
            self.current_index = 0 if reset_index else self.edit_data["remote_data"].get("remote_index", 0) if self.edit_data["remote"] else self.edit_data.get("current_index", 0)
            return

        # print(dir)
        start_dir = self.edit_data["image_dir"] or "./"

        if not isinstance(dir, str):
            dir = QtWidgets.QFileDialog.getExistingDirectory(self, "Choose Image Dir", start_dir)
            if dir:
                self.edit_data["image_dir"] = dir
                # print(self.edit_data)
                self.save_current_state()
                # print("done")
        elif not os.path.isdir(dir):
            return

        if dir:
            self.files_list.clear()
            self.files_dock_widget.listWidget.clear()

            files = []
            suffixs = tuple(['{}'.format(fmt.data().decode('ascii').lower()) for fmt in QtGui.QImageReader.supportedImageFormats()])

            for suffix in suffixs:
                files.extend(glob(os.path.join(dir, f"*.{suffix}").replace("\\", "/")))
            # for f in glob(os.path.join(dir, "*.*")):
            #     if f.lower().endswith(suffixs):
            #         # f = os.path.join(dir, f)
            #         files.append(f)
            files = sorted(files)
            self.files_list = files

            self.files_dock_widget.update_widget()

        self.current_index = 0 if reset_index else self.edit_data.get("current_index", 0)

        self.image_root = dir
        self.actionOpen_dir.setStatusTip("Image root: {}".format(self.image_root))
        if self.label_root is None:
            self.label_root = dir
            self.actionSave_dir.setStatusTip("Label root: {}".format(self.label_root))

        self.show_image(self.current_index) if show else None

    def save_dir(self, dir=None, show=True):

        start_dir = self.edit_data["label_dir"] or "./"
        if not isinstance(dir, str):
            dir = QtWidgets.QFileDialog.getExistingDirectory(self, "Choose Label Dir", start_dir)
            if dir:
                self.edit_data["label_dir"] = dir
                self.save_current_state()
        elif not os.path.isdir(dir):
            return

        if dir:
            self.label_root = dir
            self.actionSave_dir.setStatusTip("Label root: {}".format(self.label_root))

        # 刷新图片
        if self.current_index is not None and show:
            self.show_image(self.current_index)

    def save(self):
        if self.current_label is None:
            return
        self.current_label.objects.clear()
        for polygon in self.polygons:
            object = polygon.to_object()
            self.current_label.objects.append(object)

        self.current_label.note = self.info_dock_widget.lineEdit_note.text()

        if self.edit_data["remote"]:
            anno_data = json.dumps(self.current_label.to_dict())

            success, reason = remote.save_label(
                name=self.current_label.img_name,
                label_data=anno_data,
                **self.edit_data["remote_data"]
            )
            if not success and reason is not None:
                self.heart_beat_timer.stop()
                QtWidgets.QMessageBox.warning(self, 'Error', reason)
                self.heart_beat_timer.start(2000)
            else:
                self.set_saved_state(True)
        else:
            self.current_label.save_annotation()
            self.set_saved_state(True)

    def show_image(self, index: int):

        if self.edit_data["remote"]:
            if -1 < index < len(self.files_list):
                file_path = self.files_list[index]
                success, reason = remote.heart_beat(name=file_path, **self.edit_data["remote_data"])
                if not success:
                    self.heart_beat_timer.stop()
                    result = QtWidgets.QMessageBox.question(
                        self,
                        'Warning',
                        reason + "\n是否仍然尝试加载？",
                        QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
                        QtWidgets.QMessageBox.StandardButton.No
                    )
                    self.heart_beat_timer.start(2000)
                    if result == QtWidgets.QMessageBox.StandardButton.No:
                        return False

        self.reset_action()
        self.current_label = None
        self.load_finished = False
        self.saved = True
        if not -1 < index < len(self.files_list):
            self.scene.clear()
            self.scene.setSceneRect(QtCore.QRectF())
            return
        try:
            self.polygons.clear()
            self.labels_dock_widget.listWidget.clear()
            self.scene.cancel_draw()

            if self.edit_data["remote"]:
                file_path = self.files_list[index]

                success, image_data = remote.get_image(name=file_path, **self.edit_data["remote_data"])
                if not success:
                    self.heart_beat_timer.stop()
                    QtWidgets.QMessageBox.warning(self, 'Error', image_data)
                    self.heart_beat_timer.start(2000)
                    return False
            else:
                file_path = os.path.join(self.image_root, os.path.basename(self.files_list[index])).replace("\\", "/")
                image_data = Image.open(file_path)

            self.png_palette = image_data.getpalette()
            if self.png_palette is not None:
                self.statusbar.showMessage('This is a label file.')
                self.actionSegment_anything.setEnabled(False)
                self.actionPolygon.setEnabled(False)
                self.actionSave.setEnabled(False)
                self.actionBit_map.setEnabled(False)
            else:
                self.actionSegment_anything.setEnabled(self.use_segment_anything)
                self.actionPolygon.setEnabled(True)
                self.actionSave.setEnabled(True)
                self.actionBit_map.setEnabled(True)

            t0 = time.time()
            self.scene.load_image(file_path, image_data)
            # print(f"load image time: {time.time()-t0}s")
            self.view.zoomfit()

            # load label
            if self.png_palette is None:
                if self.edit_data["remote"]:
                    label_path = file_path.split(".")[0] + ".json"
                    label_data = remote.get_label(name=file_path, **self.edit_data["remote_data"])

                    # print("load remote label")
                    self.current_label = Annotation(file_path, label_path, image_data, True)
                    # print("load remote label from dict")
                    if label_data is not None:
                        try:
                            self.current_label.load_from_dict(label_data)
                        except:
                            pass
                    # print("end load remote label")

                else:
                    _, name = os.path.split(file_path)
                    label_path = os.path.join(self.label_root, '.'.join(name.split('.')[:-1]) + '.json').replace("\\", "/")
                    self.current_label = Annotation(file_path, label_path)
                    # 载入数据
                    self.current_label.load_annotation()



                last_id = 0
                for object in self.current_label.objects:
                    if object.category == self.category_choice_widget.last_category:
                        if isinstance(object.group, str):
                            if object.group.isdigit():
                                object.group = int(object.group)

                        if isinstance(object.group, int) and object.group > last_id:
                            last_id = object.group

                    polygon = Polygon()
                    self.scene.addItem(polygon)
                    polygon.load_object(object)
                    self.polygons.append(polygon)

                self.category_choice_widget.last_id = last_id

            if self.current_label is not None:
                self.setWindowTitle(f'{DEFAULT_TITLE+self.title_with_remote()} - {self.current_label.label_path}')
            else:
                self.setWindowTitle(f'{DEFAULT_TITLE+self.title_with_remote()} - {file_path}')

            self.labels_dock_widget.update_listwidget()
            self.info_dock_widget.update_widget()
            self.files_dock_widget.set_select(index)
            self.current_index = index
            if self.edit_data["remote"]:
                self.edit_data["remote_data"]["remote_index"] = self.current_index
            else:
                self.edit_data["current_index"] = self.current_index
            self.save_current_state()
            self.files_dock_widget.label_current.setText('{}'.format(self.current_index+1))
            self.load_finished = True

        except Exception as e:
            print("ERROR:", e)
            raise
        finally:
            return
            # if self.current_index > 0:
            #     self.actionPrev.setEnabled(True)
            # else:
            #     self.actionPrev.setEnabled(False)
            #
            # if self.current_index < len(self.files_list) - 1:
            #     self.actionNext.setEnabled(True)
            # else:
            #     self.actionNext.setEnabled(False)

    def prev_image(self):
        if self.scene.mode != STATUSMode.VIEW:
            return
        if self.current_index is None:
            return

        idx_before_change = self.current_index

        if not self.saved:
            self.heart_beat_timer.stop()
            result = QtWidgets.QMessageBox.question(self, 'Warning', 'Proceed without saved?', QtWidgets.QMessageBox.StandardButton.Yes|QtWidgets.QMessageBox.StandardButton.No, QtWidgets.QMessageBox.StandardButton.No)
            self.heart_beat_timer.start(2000)
            if result == QtWidgets.QMessageBox.StandardButton.No:
                return
        self.current_index = self.current_index - 1
        if self.current_index < 0:
            self.current_index = 0
            self.heart_beat_timer.stop()
            result = QtWidgets.QMessageBox.question(
                self,
                'Warning',
                'This is the first picture. Jump to the last picture?',
                QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
                QtWidgets.QMessageBox.StandardButton.No
            )
            self.heart_beat_timer.start(2000)
            if result == QtWidgets.QMessageBox.StandardButton.No:
                return
            self.current_index = len(self.files_list) - 1
        flag = self.show_image(self.current_index)
        if isinstance(flag, bool) and not flag:
            self.current_index = idx_before_change

    def next_image(self):

        # print(1, self.current_index, len(self.files_list))
        if self.scene.mode != STATUSMode.VIEW:
            return
        if self.current_index is None:
            return

        idx_before_change = self.current_index

        if not self.saved:
            self.heart_beat_timer.stop()
            result = QtWidgets.QMessageBox.question(
                self,
                'Warning',
                'Proceed without saved?',
                QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
                QtWidgets.QMessageBox.StandardButton.No
            )
            self.heart_beat_timer.start(2000)
            if result == QtWidgets.QMessageBox.StandardButton.No:
                return
        self.current_index = self.current_index + 1

        # print(2, self.current_index, len(self.files_list))

        if self.current_index > len(self.files_list)-1 and len(self.files_list) > 1:
            self.heart_beat_timer.stop()
            result = QtWidgets.QMessageBox.question(
                self,
                'Warning',
                'This is the last picture. Back to the first picture?',
                QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
                QtWidgets.QMessageBox.StandardButton.No
            )
            self.heart_beat_timer.start(2000)
            if result == QtWidgets.QMessageBox.StandardButton.No:
                return
            self.current_index = 0
        flag = self.show_image(self.current_index)
        if isinstance(flag, bool) and not flag:
            self.current_index = idx_before_change

    def jump_to(self):
        index = self.files_dock_widget.lineEdit_jump.text()
        if index:
            if not index.isdigit():
                if index in self.files_list:
                    index = self.files_list.index(index)+1
                else:
                    self.heart_beat_timer.stop()
                    QtWidgets.QMessageBox.warning(self, 'Warning', 'Don`t exist image named: {}'.format(index))
                    self.heart_beat_timer.start(2000)
                    self.files_dock_widget.lineEdit_jump.clear()
                    return
            index = int(index)-1
            if 0 <= index < len(self.files_list):
                self.show_image(index)
                self.files_dock_widget.lineEdit_jump.clear()
            else:
                self.heart_beat_timer.stop()
                QtWidgets.QMessageBox.warning(self, 'Warning', 'Index must be in [1, {}].'.format(len(self.files_list)))
                self.heart_beat_timer.start(2000)
                self.files_dock_widget.lineEdit_jump.clear()
                self.files_dock_widget.lineEdit_jump.clearFocus()
                return

    def cancel_draw(self):
        self.scene.cancel_draw()

    def setting(self):
        self.setting_dialog.load_cfg()
        self.setting_dialog.show()

    def add_new_object(self, category, group, segmentation, area, layer, bbox):
        if self.current_label is None:
            return
        object = Object(category=category, group=group, segmentation=segmentation, area=area, layer=layer, bbox=bbox)
        self.current_label.objects.append(object)

    def delete_object(self, index:int):
        if 0 <= index < len(self.current_label.objects):
            del self.current_label.objects[index]

    def change_bit_map(self):
        self.set_labels_visible(True)
        if self.scene.mode == STATUSMode.CREATE:
            self.scene.cancel_draw()
        if self.map_mode == MAPMode.LABEL:
            # to semantic
            for polygon in self.polygons:
                polygon.setEnabled(False)
                for vertex in polygon.vertexs:
                    vertex.setVisible(False)
                polygon.change_color(QtGui.QColor(self.category_color_dict.get(polygon.category, '#000000')))
                polygon.color.setAlpha(255)
                polygon.setBrush(polygon.color)
            self.labels_dock_widget.listWidget.setEnabled(False)
            self.labels_dock_widget.checkBox_visible.setEnabled(False)
            self.actionSegment_anything.setEnabled(False)
            self.actionPolygon.setEnabled(False)
            self.actionVisible.setEnabled(False)
            self.map_mode = MAPMode.SEMANTIC
            semantic_icon = QtGui.QIcon()
            semantic_icon.addPixmap(QtGui.QPixmap(":/icon/icons/semantic.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
            self.actionBit_map.setIcon(semantic_icon)

        elif self.map_mode == MAPMode.SEMANTIC:
            # to instance
            for polygon in self.polygons:
                polygon.setEnabled(False)
                for vertex in polygon.vertexs:
                    vertex.setVisible(False)
                if polygon.group != '':
                    rgb = self.instance_cmap[int(polygon.group)]
                else:
                    rgb = self.instance_cmap[0]
                polygon.change_color(QtGui.QColor(rgb[0], rgb[1], rgb[2], 255))
                polygon.color.setAlpha(255)
                polygon.setBrush(polygon.color)
            self.labels_dock_widget.listWidget.setEnabled(False)
            self.labels_dock_widget.checkBox_visible.setEnabled(False)
            self.actionSegment_anything.setEnabled(False)
            self.actionPolygon.setEnabled(False)
            self.actionVisible.setEnabled(False)
            self.map_mode = MAPMode.INSTANCE
            instance_icon = QtGui.QIcon()
            instance_icon.addPixmap(QtGui.QPixmap(":/icon/icons/instance.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
            self.actionBit_map.setIcon(instance_icon)

        elif self.map_mode == MAPMode.INSTANCE:
            # to label
            for polygon in self.polygons:
                polygon.setEnabled(True)
                for vertex in polygon.vertexs:
                    # vertex.setEnabled(True)
                    vertex.setVisible(polygon.isVisible())
                polygon.change_color(QtGui.QColor(self.category_color_dict.get(polygon.category, '#000000')))
                polygon.color.setAlpha(polygon.nohover_alpha)
                polygon.setBrush(polygon.color)
            self.labels_dock_widget.listWidget.setEnabled(True)
            self.labels_dock_widget.checkBox_visible.setEnabled(True)
            self.actionSegment_anything.setEnabled(self.use_segment_anything)
            self.actionPolygon.setEnabled(True)
            self.actionVisible.setEnabled(True)
            self.map_mode = MAPMode.LABEL
            label_icon = QtGui.QIcon()
            label_icon.addPixmap(QtGui.QPixmap(":/icon/icons/照片_pic.svg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
            self.actionBit_map.setIcon(label_icon)
        else:
            pass

    def set_labels_visible(self, visible=None, single=False):

        def try_check_single():

            selected_item = self.scene.selectedItems()
            if len(selected_item):
                self.selected_item = selected_item[0]
            else:
                pass
            selected_idx = -1
            for idx, k in enumerate(self.labels_dock_widget.polygon_item_dict.keys()):
                # print(k, self.scene.selectedItems()[0])
                if k == self.selected_item:
                    selected_idx = idx
                # print(self.selected_item, k)
            if selected_idx == -1:
                selected_idx = self.labels_dock_widget.listWidget.currentRow()
            # print(self.selected_item, selected_idx)

            for idx, check_box in enumerate(self.labels_dock_widget.check_boxes):
                check_box.setChecked(idx == selected_idx or not self.single_vis)

        if single:
            self.single_vis = not self.single_vis
            try:
                try_check_single()
            except Exception as e:
                self.single_vis = not self.single_vis
                print(e)
            return

        if visible is None:
            visible = not self.labels_dock_widget.checkBox_visible.isChecked()
            self.labels_dock_widget.checkBox_visible.setChecked(visible)

        self.labels_dock_widget.set_all_polygon_visible(visible)

        if self.single_vis and (visible or False):
            # print("single")
            try_check_single()

    def label_converter(self):
        self.convert_dialog.reset_gui()
        self.convert_dialog.show()

    def convert_coco(self):
        self.convert_coco_win.COCODistPath.clear()
        self.convert_coco_win.COCOAnnoPath.clear()
        self.convert_coco_win.image_path.setText(self.edit_data["image_dir"])
        self.convert_coco_win.label_path.setText(self.edit_data["label_dir"])
        self.convert_coco_win.open()

    def help(self):
        self.shortcut_dialog.show()

    def about(self):
        self.about_dialog.show()

    def save_cfg(self, config_file):
        save_config(self.cfg, config_file)

    def exit(self):
        self.save_cfg(self.config_file)
        self.close()

    def closeEvent(self, a0: QtGui.QCloseEvent):
        self.heart_beat_timer.stop()
        self.exit()

    def init_connect(self):
        self.actionOpen_dir.triggered.connect(self.open_dir)
        self.actionSave_dir.triggered.connect(self.save_dir)
        self.actionPrev.triggered.connect(self.prev_image)
        self.actionNext.triggered.connect(self.next_image)
        self.actionSetting.triggered.connect(self.setting)
        self.actionExit.triggered.connect(self.exit)
        self.actionRemoteSetting.triggered.connect(self.remote_settings_win.show)

        self.actionSegment_anything.triggered.connect(self.scene.start_segment_anything)
        self.actionPolygon.triggered.connect(self.scene.start_draw_polygon)
        self.actionCancel.triggered.connect(self.scene.cancel_draw)
        self.actionBackspace.triggered.connect(self.scene.backspace)
        self.actionFinish.triggered.connect(self.scene.finish_draw)
        self.actionEdit.triggered.connect(self.scene.edit_polygon)
        self.actionDelete.triggered.connect(self.scene.delete_selected_graph)
        self.actionSave.triggered.connect(self.save)
        self.actionTo_top.triggered.connect(self.scene.move_polygon_to_top)
        self.actionTo_bottom.triggered.connect(self.scene.move_polygon_to_bottom)

        self.actionZoom_in.triggered.connect(self.view.zoom_in)
        self.actionZoom_out.triggered.connect(self.view.zoom_out)
        self.actionFit_wiondow.triggered.connect(self.view.zoomfit)
        self.actionBit_map.triggered.connect(self.change_bit_map)
        self.actionVisible.triggered.connect(functools.partial(self.set_labels_visible, None))

        self.actionVisSingle.clicked.connect(functools.partial(self.set_labels_visible, None, True))

        self.actionConverter.triggered.connect(self.label_converter)
        self.actionConverter_coco.triggered.connect(self.convert_coco)

        self.actionShortcut.triggered.connect(self.help)
        self.actionAbout.triggered.connect(self.about)

        self.actionChinese.triggered.connect(self.translate_to_chinese)
        self.actionEnglish.triggered.connect(self.translate_to_english)

        self.labels_dock_widget.listWidget.doubleClicked.connect(self.scene.edit_polygon)

    def reset_action(self):
        self.actionPrev.setEnabled(False)
        self.actionNext.setEnabled(False)
        self.actionSegment_anything.setEnabled(False)
        self.actionPolygon.setEnabled(False)
        self.actionEdit.setEnabled(False)
        self.actionDelete.setEnabled(False)
        self.actionSave.setEnabled(False)
        self.actionTo_top.setEnabled(False)
        self.actionTo_bottom.setEnabled(False)
        self.actionBit_map.setChecked(False)
        self.actionBit_map.setEnabled(False)

    def __heart_beat(self):
        if self.edit_data["remote"] and isinstance(self.current_index, int):
            file_path = self.files_list[self.current_index]
            success, reason = remote.heart_beat(name=file_path, **self.edit_data["remote_data"])
            # if not success:
            #     self.heart_beat_timer.stop()
            #     QtWidgets.QMessageBox.warning(self, 'Error', reason)
            #     self.heart_beat_timer.start(2000)
