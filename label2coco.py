from PyQt5 import QtGui, QtWidgets, QtCore
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import QDialog, QFileDialog
from PyQt5.uic import loadUi


from glob import glob
from multiprocessing import Process, freeze_support, Queue
import os
import os.path as osp
import datetime
import time

# from utils.voc2coco import COCO
from utils.convert2coco import to_coco, divide_coco_by_image, COCO

from ui.labelConverter import Ui_MainWindow as labelConverter


freeze_support()


class LabelConverter(QtWidgets.QMainWindow, labelConverter):  #

    to_coco = False
    divide = False

    to_coco_Progress: Process = None
    to_coco_ProgressQueue = Queue()

    DivideProgress: Process = None
    DivideProgressQueue = Queue()

    msgs = []

    def __init__(self, parent=None):
        super().__init__()
        self.setupUi(self)
        # loadUi("ui/datasetConverter.ui", self)
        self.msg.setVisible(False)
        self.p = parent
        if self.p is not None:
            self.setWindowIcon(self.p.windowIcon())

        label = QtWidgets.QLabel('<a href="https://github.com/LSH9832">作者主页</a>')
        label.setOpenExternalLinks(True)
        self.statusbar.addPermanentWidget(label)
        # self.statusBar()

        def init_connect():

            # TO COCO
            def browseImagePath():
                try:
                    path = self.image_path.text() if os.path.isdir(self.image_path.text()) else "./"
                    ret = QFileDialog.getExistingDirectory(self, "Choose Image Path", path)
                    print(1)
                    if len(ret):
                        print(2, ret)
                        self.image_path.setText(ret.replace("\\", "/"))
                        print(3)
                except Exception as e:
                    print(e)

            self.browseImagePath.clicked.connect(browseImagePath)

            def browseLabelPath():
                path = self.label_path.text() if os.path.isdir(self.label_path.text()) else "./"
                ret = QFileDialog.getExistingDirectory(self, "Choose Label Path", path)
                if len(ret):
                    self.label_path.setText(ret.replace("\\", "/"))

            self.browseLabelPath.clicked.connect(browseLabelPath)

            def browseCOCOPath():
                path = self.COCODistPath.text() if os.path.isdir(self.COCODistPath.text()) else "./"
                ret = QFileDialog.getExistingDirectory(self, "Choose COCO Dataset Save Path", path)
                if len(ret):
                    self.COCODistPath.setText(ret.replace("\\", "/"))

            self.browseCOCODistPath.clicked.connect(browseCOCOPath)

            def to_coco_start_stop():
                if not self.to_coco:
                    # TODO 检查目标文件夹是否为空文件夹，如果不是是否覆盖还是合并续写
                    target_dir = os.path.join(self.COCODistPath.text(), self.COCODatasetName.text())

                    self.to_coco_ProgressQueue.get() if not self.to_coco_ProgressQueue.empty() else None
                    kwargs = {
                        "image_dirs": self.image_path.text(),
                        "annotation_dirs": self.label_path.text(),
                        "class_file": None,
                        "data_name": self.COCODatasetName.text(),
                        "image_target_dir": osp.join(target_dir, self.COCODistImagePath.text()),
                        "image_type": "jpg",
                        "version": self.COCOVersion.text(),
                        "url": "",
                        "author": self.COCOAuthor.text(),
                        "save": True,
                        "dist_dir": target_dir,
                        "coco_dataset": None,
                        "q": self.to_coco_ProgressQueue
                    }

                    if os.path.isdir(os.path.join(target_dir, "annotations")):
                        json_file = os.path.join(target_dir,
                                                 "annotations",
                                                 f"{self.COCODatasetName.text()}.json")
                        if os.path.isfile(json_file):
                            if QtWidgets.QMessageBox.warning(
                                self,
                                "警告",
                                f"标签文件夹下{self.COCODatasetName.text()}已存在，确定继续转换？",
                                QtWidgets.QMessageBox.Yes,
                                QtWidgets.QMessageBox.No
                            ) == QtWidgets.QMessageBox.Yes:
                                # TODO 覆盖 or 续写
                                response = QtWidgets.QMessageBox.information(
                                    self,
                                    "请选择",
                                    "覆盖（Y）或续写（N）？（关闭此对话框为N）",
                                    QtWidgets.QMessageBox.Yes,
                                    QtWidgets.QMessageBox.No,
                                )
                                if response == QtWidgets.QMessageBox.No:
                                    kwargs["coco_dataset"] = COCO(json_file)
                                elif not response == QtWidgets.QMessageBox.Yes:
                                    return
                            else:
                                return

                    os.makedirs(target_dir, exist_ok=True)
                    os.makedirs(kwargs["image_target_dir"], exist_ok=True)
                    os.makedirs(os.path.join(target_dir, "annotations"), exist_ok=True)

                    print("start voc2coco")
                    self.to_coco_Progress = Process(target=to_coco, kwargs=kwargs)
                    self.to_coco_Progress.start()

                    # self.progressBar: QtWidgets.QProgressBar()
                    self.progressBar.setMaximum(self.to_coco_ProgressQueue.get())

                    self.to_coco = True
                    self.set_to_coco_Enabled(False)
                    self.ConvertButton.setText("停止转换")
                    self.addMsg("转换开始")

                else:
                    if self.to_coco_Progress is not None:
                        self.to_coco_Progress.terminate()
                    self.to_coco = False
                    self.set_to_coco_Enabled(True)
                    self.progressBar.setValue(0)
                    self.ConvertButton.setText("开始转换")
                    self.addMsg("转换中止，需手动删除已转换的文件")

            self.ConvertButton.clicked.connect(to_coco_start_stop)

            # Divide
            def browseJsonPath():
                path = self.COCOAnnoPath.text() if osp.isfile(self.COCOAnnoPath.text()) else "./"
                ret, _ = QFileDialog.getOpenFileName(self, "Choose COCO Json File", path, "json文件 (*.json)")
                # print(len(ret))
                if len(ret):
                    self.COCOAnnoPath.setText(ret)

            self.browseCOCOAnnoPath.clicked.connect(browseJsonPath)

            def divide_start_stop():
                if not self.divide:   # 说明要开始拆分了

                    kwargs = {
                        "json_file": self.COCOAnnoPath.text(),
                        "train": self.train_ratio.value(),
                        "val": self.val_ratio.value(),
                        "q": self.DivideProgressQueue
                    }

                    print("start")
                    self.DivideProgress = Process(target=divide_coco_by_image, kwargs=kwargs)
                    self.DivideProgress.start()

                    self.divide = True
                    self.setDivideEnabled(False)
                    self.DivideButton.setText("停止拆分")
                    self.addMsg("拆分开始")

                else:                 # 说明要停止拆分了
                    if self.DivideProgress is not None:
                        self.DivideProgress.terminate()
                    self.divide = False
                    self.setDivideEnabled(True)
                    self.DivideButton.setText("开始拆分")
                    self.addMsg("拆分中止")

            self.DivideButton.clicked.connect(divide_start_stop)

        init_connect()
        self.addMsg("开始运行")

        def init_timer():

            def timeoutFun0():
                is_to_coco_complete = self.check_complete(True)
                self.ConvertButton.setEnabled(is_to_coco_complete)
                is_divide_complete = self.check_complete(False)
                self.DivideButton.setEnabled(is_divide_complete)

            self.timer0 = QTimer()
            self.timer0.timeout.connect(timeoutFun0)
            self.timer0.start(200)

            def timeoutFun1():
                # print(self.to_coco, self.divide)
                if self.to_coco:
                    if not self.to_coco_ProgressQueue.empty():
                        try:
                            per = self.to_coco_ProgressQueue.get()
                            # self.progressBar = QtWidgets.QProgressBar()
                            # print(per, type(per))

                            if self.progressBar.value() < self.progressBar.maximum():
                                self.progressBar.setValue(self.progressBar.value() + per)
                                self.update()

                            if self.progressBar.value() == self.progressBar.maximum():
                                self.to_coco = False
                                self.addMsg(self.to_coco_ProgressQueue.get())

                                self.update()

                                QtWidgets.QMessageBox.information(
                                    self,
                                    "完成",
                                    "转换完成",
                                    QtWidgets.QMessageBox.Ok
                                )
                                self.progressBar.setValue(0)
                                self.set_to_coco_Enabled(True)
                                self.ConvertButton.setText("开始转换")
                        except Exception as e:
                            print(e)
                            QtWidgets.QMessageBox.warning(
                                self,
                                "错误",
                                str(e),
                                QtWidgets.QMessageBox.Ok
                            )

                if self.divide:
                    if not self.DivideProgress.is_alive():
                        self.divide = False
                        self.addMsg(self.DivideProgressQueue.get())
                        self.update()
                        QtWidgets.QMessageBox.information(
                            self,
                            "完成",
                            "拆分完成",
                            QtWidgets.QMessageBox.Ok
                        )
                        self.setDivideEnabled(True)
                        self.ConvertButton.setText("开始拆分")


                # self.val_ratio = QtWidgets.QDoubleSpinBox()
                self.val_ratio.setMaximum(1. - self.train_ratio.value())
                self.test_ratio.setValue(1. - self.val_ratio.value() - self.train_ratio.value())

            self.timer1 = QTimer()
            self.timer1.timeout.connect(timeoutFun1)
            self.timer1.start(1)

        init_timer()


    def closeEvent(self, e):
        if self.p is not None:
            self.p.setVisible(True)

    def open(self):
        if self.p is not None:
            self.p.setVisible(False)
        self.show()

    def addMsg(self, msg: str):
        self.msgs.append(f"[{str(datetime.datetime.now()).split('.')[0]}] {msg}")
        show_str = ""
        for m in self.msgs:
            show_str += f"{m}\n"
        self.msg.setText(show_str)
        # self.msg = QtWidgets.QTextBrowser()
        self.msg.moveCursor(QtGui.QTextCursor.End)

    def set_to_coco_Enabled(self, flag: bool):
        self.image_path.setEnabled(flag)
        self.browseImagePath.setEnabled(flag)
        self.label_path.setEnabled(flag)
        self.browseLabelPath.setEnabled(flag)
        self.COCODistPath.setEnabled(flag)
        self.browseCOCODistPath.setEnabled(flag)
        self.COCODistImagePath.setEnabled(flag)
        self.COCODatasetName.setEnabled(flag)
        self.COCOAuthor.setEnabled(flag)
        self.COCOVersion.setEnabled(flag)

    def setDivideEnabled(self, flag: bool):
        self.COCOAnnoPath.setEnabled(flag)
        self.browseCOCOAnnoPath.setEnabled(flag)
        self.train_ratio.setEnabled(flag)
        self.val_ratio.setEnabled(flag)
        self.test_ratio.setEnabled(flag)

    def check_complete(self, to_coco=True):
        try:
            flag = False
            if to_coco:
                image_path = self.image_path.text()
                label_path = self.label_path.text()
                dist_path = self.COCODistPath.text()
                dist_image_path = self.COCODistImagePath.text()
                img_dir = self.COCODatasetName.text()

                check_strings = [image_path, label_path, dist_path, dist_image_path, img_dir]

                if all([len(s) > 0 for s in check_strings]):
                    if osp.isdir(image_path) and osp.isdir(label_path):
                        if len(self.COCODatasetName.text()) * len(self.COCOAuthor.text()) * len(self.COCOVersion.text()):
                            flag = True
            elif len(self.COCOAnnoPath.text()):
                flag = True
        except Exception as e:
            print(e)
            return False
        return flag

    def closeEvent(self, a0: QtGui.QCloseEvent):
        if self.p is not None:
            self.p.show()


def main():
    import sys

    app = QtWidgets.QApplication(sys.argv)

    w = LabelConverter()
    w.show()

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
