from PyQt5 import QtGui, QtWidgets, QtCore
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import QDialog, QFileDialog
from PyQt5.uic import loadUi


from glob import glob, iglob
from multiprocessing import Process, freeze_support, Queue
import os
import os.path as osp
import datetime
import time
import platform
# from utils.voc2coco import COCO
from utils.convert2coco import to_coco, divide_coco_by_image, COCO, coco2yolo

from ui.labelConverter import Ui_MainWindow as labelConverter
from typing import List


freeze_support()


class LabelConverter(QtWidgets.QMainWindow, labelConverter):  #

    to_coco = False
    divide = False
    coco2yolo = False

    to_coco_Progress: Process = None
    to_coco_ProgressQueue = Queue()

    DivideProgress: Process = None
    DivideProgressQueue = Queue()

    gen_idx = False
    gen_idx_processes: List[Process] = []
    gen_idx_queue = Queue()
    
    coco2yoloProgresses: List[Process] = []
    coco2yoloQueue = Queue()
    coco2yoloPercent = [0., 0., 0.]
    coco2yoloNum = [0, 0, 0]


    msgs = []

    def __init__(self, parent=None):
        super().__init__()
        self.setupUi(self)
        loadUi("ui/datasetConverter.ui", self)
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

            def browseCOCOPath(editors, title="Choose COCO Dataset Path", relpath=None, force_start_path=None):
                if not isinstance(editors, list):
                    editors = [editors]
                path = force_start_path if force_start_path is not None else editors[0].text() if os.path.isdir(editors[0].text()) else "./"
                ret = QFileDialog.getExistingDirectory(self, title, path)
                if len(ret):
                    if relpath is not None:
                        ret = osp.relpath(ret, relpath)
                    for editor in editors:
                        editor.setText(ret.replace("\\", "/"))
                    

            self.browseCOCODistPath.clicked.connect(lambda: browseCOCOPath([self.COCODistPath, self.COCODatasetPath]))
            self.browseCOCODatasetPath.clicked.connect(lambda: browseCOCOPath(self.COCODatasetPath))
            self.browseCurrentImgPath.clicked.connect(lambda: browseCOCOPath(self.currentimg_path, "Choose Current Image Path", self.COCODatasetPath.text(), self.COCODatasetPath.text()))
            self.browseTrainImgPath.clicked.connect(lambda: browseCOCOPath(self.trainimg_path, "Choose Train Image Path", self.COCODatasetPath.text(), self.COCODatasetPath.text()))
            self.browseValImgPath.clicked.connect(lambda: browseCOCOPath(self.valimg_path, "Choose Val Image Path", self.COCODatasetPath.text(), self.COCODatasetPath.text()))
            self.browseTestImgPath.clicked.connect(lambda: browseCOCOPath(self.testimg_path, "Choose Test Image Path", self.COCODatasetPath.text(), self.COCODatasetPath.text()))
            
            def fill_label():
                coco_root_path = self.COCODatasetPath.text()
                path_valid = len(coco_root_path) and osp.isdir(coco_root_path)
                self.divideImgPath.setEnabled(path_valid)
                self.divideImgPath.setChecked(path_valid and self.divideImgPath.isChecked())
                self.currentimg_path.setEnabled(path_valid)
                self.browseCurrentImgPath.setEnabled(path_valid)
                self.generateIndexFile.setEnabled(path_valid)
                self.startConvertButton.setEnabled(path_valid)

                self.oriCOCOTrainLabelFile.clear()
                self.oriCOCOValLabelFile.clear()
                self.oriCOCOTestLabelFile.clear()

                if path_valid:

                    train_idx = -1
                    val_idx = -1
                    test_idx = -1

                    

                    self.oriCOCOTrainLabelFile.addItem("-")
                    self.oriCOCOValLabelFile.addItem("-")
                    self.oriCOCOTestLabelFile.addItem("-")
                    
                    for i, anno_file in enumerate(glob(osp.join(coco_root_path, "annotations", "*.json"))):
                        anno_file = osp.basename(anno_file)
                        self.oriCOCOTrainLabelFile.addItem(anno_file)
                        self.oriCOCOValLabelFile.addItem(anno_file)
                        self.oriCOCOTestLabelFile.addItem(anno_file)
                        if "train" in anno_file.lower() and (train_idx < 0 or "instances" in anno_file.lower()):
                            self.oriCOCOTrainLabelFile.setCurrentIndex(i+1)
                            train_idx = i
                        elif "val" in anno_file.lower() and (val_idx < 0 or "instances" in anno_file.lower()):
                            self.oriCOCOValLabelFile.setCurrentIndex(i+1)
                            val_idx = i
                        elif "test" in anno_file.lower() and (test_idx < 0  or "instances" in anno_file.lower()):
                            self.oriCOCOTestLabelFile.setCurrentIndex(i+1)
                            test_idx = i

                    # if self.convertTestLabels.isChecked():
                    #     pass
                
            
            self.oriCOCOTestLabelFile.setVisible(False)
            self.YOLOTestLabelPath.setVisible(False)
            self.label_7.setVisible(False)
            self.label_10.setVisible(False)
            self.testimg_path.setVisible(False)
            self.browseTestImgPath.setVisible(False)
            self.COCODatasetPath.textChanged.connect(fill_label)

            self.limitSuffix.clicked.connect(lambda: self.imgSuffix.setEnabled(self.limitSuffix.isChecked()))

            self.COCODistImagePath.textChanged.connect(lambda: self.currentimg_path.setText(self.COCODistImagePath.text()))
            # self.convertTestLabels.clicked.connect(lambda: self.oriCOCOTestLabelFile.setEnabled(self.convertTestLabels.isChecked()))
            self.convertTestLabels.clicked.connect(lambda: self.oriCOCOTestLabelFile.setVisible(self.convertTestLabels.isChecked()))
            self.convertTestLabels.clicked.connect(lambda: self.YOLOTestLabelPath.setVisible(self.convertTestLabels.isChecked()))
            self.convertTestLabels.clicked.connect(lambda: self.label_7.setVisible(self.convertTestLabels.isChecked()))
            self.convertTestLabels.clicked.connect(lambda: self.testimg_path.setVisible(self.convertTestLabels.isChecked()))
            self.convertTestLabels.clicked.connect(lambda: self.label_10.setVisible(self.convertTestLabels.isChecked()))
            self.convertTestLabels.clicked.connect(lambda: self.browseTestImgPath.setVisible(self.convertTestLabels.isChecked()))

            
            self.enableCustomPath.clicked.connect(lambda: self.YOLOTrainLabelPath.setEnabled(self.enableCustomPath.isChecked()))
            self.enableCustomPath.clicked.connect(lambda: self.YOLOValLabelPath.setEnabled(self.enableCustomPath.isChecked()))
            self.enableCustomPath.clicked.connect(lambda: self.YOLOTestLabelPath.setEnabled(self.enableCustomPath.isChecked()))
            
            self.divideImgPath.clicked.connect(lambda: self.trainimg_path.setEnabled(self.divideImgPath.isChecked()))
            self.divideImgPath.clicked.connect(lambda: self.valimg_path.setEnabled(self.divideImgPath.isChecked()))
            self.divideImgPath.clicked.connect(lambda: self.testimg_path.setEnabled(self.divideImgPath.isChecked()))
            self.divideImgPath.clicked.connect(lambda: self.browseTrainImgPath.setEnabled(self.divideImgPath.isChecked()))
            self.divideImgPath.clicked.connect(lambda: self.browseValImgPath.setEnabled(self.divideImgPath.isChecked()))
            self.divideImgPath.clicked.connect(lambda: self.browseTestImgPath.setEnabled(self.divideImgPath.isChecked()))
            
            def gen_label_idx():

                coco_root_path = self.COCODatasetPath.text()

                def gen_once(img_path, save_file, suffix="*", q=None):
                    if q is not None:
                        q.put(f"generating {save_file}, please wait")
                    if not osp.isdir(osp.join(coco_root_path, img_path)):
                        return
                    savestr = ""
                    print(osp.join(coco_root_path, img_path, f"*.{suffix}"))
                    for imgf in iglob(osp.join(coco_root_path, img_path, f"*.{suffix}")):
                        savestr += osp.relpath(imgf, coco_root_path).replace("\\", "/") + "\n"
                    os.makedirs(osp.dirname(save_file), exist_ok=True)
                    with open(save_file, "w") as wf:
                        wf.write(savestr[:-1])
                    if q is not None:
                        q.put(f"write {save_file} done")
                        
                
                
                if "windows" in platform.platform().lower():
                    imgsuffix = self.imgSuffix.text() if self.limitSuffix.isChecked() else "*"
                    if len(self.trainimg_path.text()):
                        gen_once(
                            self.trainimg_path.text(), 
                            osp.join(coco_root_path, "labels/train.txt"), 
                            imgsuffix,
                            self.gen_idx_queue
                        )
                    if len(self.valimg_path.text()):
                        gen_once(
                            self.valimg_path.text(), 
                            osp.join(coco_root_path, "labels/val.txt"), 
                            imgsuffix,
                            self.gen_idx_queue
                        )
                    if len(self.testimg_path.text()) and self.convertTestLabels.isChecked():
                        gen_once(
                            self.testimg_path.text(), 
                            osp.join(coco_root_path, "labels/test.txt"),
                            imgsuffix,
                            self.gen_idx_queue
                        )
                    self.gen_idx = True
                    return
                
                if not self.gen_idx:
                    imgsuffix = self.imgSuffix.text() if self.limitSuffix.isChecked() else "*"
                    self.gen_idx_processes.clear()
                    if len(self.trainimg_path.text()):
                        self.gen_idx_processes.append(
                            Process(
                                target=gen_once, 
                                args=(
                                    self.trainimg_path.text(), 
                                    osp.join(coco_root_path, "labels/train.txt"), 
                                    imgsuffix,
                                    self.gen_idx_queue
                                )
                            )
                        )
                    if len(self.valimg_path.text()):
                        self.gen_idx_processes.append(
                            Process(
                                target=gen_once, 
                                args=(
                                    self.valimg_path.text(), 
                                    osp.join(coco_root_path, "labels/val.txt"), 
                                    imgsuffix,
                                    self.gen_idx_queue
                                )
                            )
                        )
                    if len(self.testimg_path.text()) and self.convertTestLabels.isChecked():
                        self.gen_idx_processes.append(
                            Process(
                                target=gen_once, 
                                args=(
                                    self.testimg_path.text(), 
                                    osp.join(coco_root_path, "labels/test.txt"),
                                    imgsuffix,
                                    self.gen_idx_queue
                                )
                            )
                        )
                    [process.start() for process in self.gen_idx_processes]
                    self.gen_idx = True
                    self.set_coco2yolo_Enabled(False)

            self.generateIndexFile.clicked.connect(gen_label_idx)
            
            def coco2yolo_start_stop():
                if not self.coco2yolo:
                    while not self.coco2yoloQueue.empty():
                        self.coco2yoloQueue.get()

                    coco_root_path = self.COCODatasetPath.text()
                    
                    train_labels_file = osp.join(
                        coco_root_path,
                        "annotations",
                        self.oriCOCOTrainLabelFile.currentText()
                    )
                    val_labels_file = osp.join(
                        coco_root_path,
                        "annotations",
                        self.oriCOCOValLabelFile.currentText()
                    )

                    test_labels_file = osp.join(
                        coco_root_path,
                        "annotations",
                        self.oriCOCOTestLabelFile.currentText()
                    )

                    task_files = []
                    img_dist_dirs = []
                    if osp.isfile(train_labels_file):
                        task_files.append((train_labels_file, osp.join(coco_root_path, self.YOLOTrainLabelPath.text())))
                        img_dist_dirs.append(self.trainimg_path.text())
                    if osp.isfile(val_labels_file):
                        task_files.append((val_labels_file, osp.join(coco_root_path, self.YOLOValLabelPath.text())))
                        img_dist_dirs.append(self.valimg_path.text())
                    if self.convertTestLabels.isChecked() and osp.isfile(test_labels_file):
                        task_files.append((test_labels_file, osp.join(coco_root_path, self.YOLOTestLabelPath.text())))
                        img_dist_dirs.append(self.testimg_path.text())
                    
                    self.coco2yoloProgresses.clear()
                    for i, file_paths in enumerate(task_files):
                        self.coco2yoloProgresses.append(
                            Process(
                                target=coco2yolo, 
                                args=file_paths, 
                                kwargs={
                                    "moveImg": self.divideImgPath.isChecked() and len(self.currentimg_path.text()) and len(img_dist_dirs[i]),
                                    "oriImgPath": osp.join(coco_root_path, self.currentimg_path.text()),
                                    "distImgPath": osp.join(coco_root_path, img_dist_dirs[i]),
                                    "q": self.coco2yoloQueue, 
                                    "num_process": len(task_files), 
                                    "rank": i
                                }
                            )
                        )
                    
                    self.coco2yoloPercent = [0., 0., 0.]
                    self.coco2yoloNum = [0, 0, 0]
                    self.progressBar_2.setMaximum(1000)
                    [process.start() for process in self.coco2yoloProgresses]
                    self.coco2yolo = True
                    self.set_coco2yolo_Enabled(False)
                    self.startConvertButton.setText("停止转换")
                    self.addMsg("转换开始")
                else:
                    [process.terminate() for process in self.coco2yoloProgresses]
                    self.coco2yolo = False
                    self.set_coco2yolo_Enabled(True)
                    self.progressBar_2.setValue(0)
                    self.startConvertButton.setText("开始转换")


            self.startConvertButton.clicked.connect(coco2yolo_start_stop)

            def to_coco_start_stop():
                if not self.to_coco:
                    # TODO 检查目标文件夹是否为空文件夹，如果不是是否覆盖还是合并续写
                    target_dir = os.path.join(self.COCODistPath.text(), self.COCODatasetName.text())

                    self.to_coco_ProgressQueue.get() if not self.to_coco_ProgressQueue.empty() else None

                    label_path = self.label_path.text()
                    txt_file_list = glob(osp.join(osp.dirname(label_path), "**/*.txt").replace("\\", "/"), recursive=True)
                    self.addMsg(f"use class file {txt_file_list[0]}" if len(txt_file_list) else "no txt class file found")

                    kwargs = {
                        "image_dirs": self.image_path.text(),
                        "annotation_dirs": label_path,
                        "class_file": txt_file_list[0] if len(txt_file_list) else None,
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
                                while not self.to_coco_ProgressQueue.empty():
                                    self.addMsg(self.to_coco_ProgressQueue.get())
                                try:
                                    self.to_coco_Progress.terminate()
                                except:
                                    pass
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
                        while not self.DivideProgressQueue.empty():
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
                
                if self.gen_idx:
                    if not any([process.is_alive() for process in self.gen_idx_processes]):
                        self.gen_idx = False
                        while not self.gen_idx_queue.empty():
                            data = self.gen_idx_queue.get()
                            if isinstance(data, str):
                                self.addMsg(data)
                        QtWidgets.QMessageBox.information(
                            self,
                            "完成",
                            "索引生成完成",
                            QtWidgets.QMessageBox.Ok
                        )
                        self.set_coco2yolo_Enabled(True)
                    else:
                        if not self.gen_idx_queue.empty():
                            data = self.gen_idx_queue.get()
                            if isinstance(data, str):
                                self.addMsg(data)
                
                if self.coco2yolo:
                    if not any([process.is_alive() for process in self.coco2yoloProgresses]):
                        self.coco2yolo = False
                        self.set_coco2yolo_Enabled(True)
                        self.startConvertButton.setText("开始转换")
                        while not self.coco2yoloQueue.empty():
                            data = self.coco2yoloQueue.get()
                            if isinstance(data, str):
                                self.addMsg(data)
                        self.progressBar_2.setValue(1000)
                        QtWidgets.QMessageBox.information(
                            self,
                            "完成",
                            "转换完成",
                            QtWidgets.QMessageBox.Ok
                        )
                        self.progressBar_2.setValue(0)
                    else:
                        if not self.coco2yoloQueue.empty():
                            data = self.coco2yoloQueue.get()
                            if isinstance(data, str):
                                self.addMsg(data)
                            if isinstance(data, list):
                                
                                self.coco2yoloNum[data[0]] = data[2]
                                for idx in range(len(self.coco2yoloProgresses)):
                                    if self.coco2yoloNum[idx] == 0:
                                        return
                                self.coco2yoloPercent[data[0]] = data[1] * (data[2] / sum(self.coco2yoloNum))
                                value = int(sum(self.coco2yoloPercent) * 1000)
                                self.progressBar_2.setValue(value)
                                if value == 1000:
                                    [process.terminate() for process in self.coco2yoloProgresses]
                                    



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
    
    def set_coco2yolo_Enabled(self, flag: bool):
        self.COCODatasetPath.setEnabled(flag)
        self.browseCOCODatasetPath.setEnabled(flag)
        self.oriCOCOTrainLabelFile.setEnabled(flag)
        self.oriCOCOValLabelFile.setEnabled(flag)
        self.oriCOCOTestLabelFile.setEnabled(flag)
        self.convertTestLabels.setEnabled(flag)
        self.enableCustomPath.setEnabled(flag)
        self.generateIndexFile.setEnabled(flag)
        self.YOLOTrainLabelPath.setEnabled(flag and self.enableCustomPath.isChecked())
        self.YOLOValLabelPath.setEnabled(flag and self.enableCustomPath.isChecked())
        self.YOLOTestLabelPath.setEnabled(flag and self.enableCustomPath.isChecked())

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
