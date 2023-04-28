from PyQt5.uic import loadUi
from PyQt5 import QtGui, QtWidgets
from PyQt5.QtCore import *

import os
import os.path as osp
import yaml
import utils.remote as remote


class RemoteSettings(QtWidgets.QMainWindow):  # , labelConverter

    connected = False
    remote_data = None
    use_remote = False

    def __init__(self, parent=None):
        super().__init__()
        # self.setupUi(self)
        self.parent = parent
        loadUi(f"{'.' * (2 if self.parent is None else 1)}/ui/remote_settings.ui", self)

        # self.remote_mode_enabled = QtWidgets.QCheckBox()
        # self.password = QtWidgets.QLineEdit()
        self.show_passwd.clicked.connect(self.__show_passwd)
        self.test_connect.clicked.connect(self.try_connect)
        self.cancel.clicked.connect(self.close)
        self.confirm.clicked.connect(self.apply)

        self.load_current_settings()
        self.success_data = None

        self.timer = QTimer()
        self.timer.timeout.connect(self.check__)
        # self.timer.start(50)

    def showEvent(self, a0: QtGui.QShowEvent) -> None:
        self.load_current_settings()
        self.connect_result.setStyleSheet("QLabel{background-color: rgb(255, 0, 0);}")
        self.connect_result.setText("disconnected")
        self.connected = False
        self.success_data = None
        self.timer.start(50)

    def closeEvent(self, a0: QtGui.QCloseEvent) -> None:
        self.timer.stop()

    def apply(self):
        if self.parent is not None:
            self.parent.edit_data["remote"] = self.remote_mode_enabled.isChecked()
            for k, v in self.to_dict().items():
                self.parent.edit_data["remote_data"][k] = v
            self.parent.save_current_state()
            print(self.parent.edit_data["remote_data"])

            self.parent.reload_mode()
        else:
            if osp.isfile("../settings/last_edit.yaml"):
                data = yaml.load(open("../settings/last_edit.yaml"), yaml.SafeLoader)
                data["remote"] = self.remote_mode_enabled.isChecked()
                data["remote_data"] = self.to_dict()
                with open("../settings/last_edit.yaml", "w") as f:
                    yaml.dump(data, f, yaml.Dumper)
        self.close()

    def check__(self):
        if self.success_data is not None:
            if not self.success_data == self.to_dict():
                self.connect_result.setStyleSheet("QLabel{background-color: rgb(255, 0, 0);}")
                self.connect_result.setText("disconnected")
                self.connected = False

        self.confirm.setEnabled(self.connected or not self.remote_mode_enabled.isChecked())

    def to_dict(self):
        return {
            "ip": self.ip.text(),
            "port": self.port.value(),
            "user": self.user.text(),
            "passwd": self.password.text()
        }

    def try_connect(self):
        success, reason = remote.test_connect(**self.to_dict())
        self.connected = success
        if success:
            self.connect_result.setStyleSheet("QLabel{background-color: rgb(10, 255, 10);}")
            self.connect_result.setText("connected")
            self.success_data = self.to_dict()
        else:
            self.connect_result.setStyleSheet("QLabel{background-color: rgb(255, 0, 0);}")
            self.connect_result.setText("disconnected")
            QtWidgets.QMessageBox.warning(self, 'Connect Error', reason)

    def load_current_settings(self):
        self.connected = False
        self.use_remote = False
        if self.parent is not None:
            self.setWindowIcon(self.parent.windowIcon())
            self.remote_data = self.parent.edit_data["remote_data"]
            self.use_remote = self.parent.edit_data["remote"]
        else:
            if osp.isfile("../settings/last_edit.yaml"):
                data = yaml.load(open("../settings/last_edit.yaml"), yaml.SafeLoader)
                self.remote_data = data["remote_data"]
                self.use_remote = data["remote"]

        self.ip.setText(self.remote_data["ip"])
        self.port.setValue(self.remote_data["port"])
        self.user.setText(str(self.remote_data["user"]))
        self.password.setText(str(self.remote_data["passwd"]))
        self.remote_mode_enabled.setChecked(self.use_remote)

    def __show_passwd(self):
        try:
            self.password.setEchoMode(QtWidgets.QLineEdit.Normal if self.show_passwd.isChecked() else QtWidgets.QLineEdit.Password)
        except Exception as e:
            print(e)


def main():
    import sys
    app = QtWidgets.QApplication(sys.argv)
    w = RemoteSettings()
    w.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
