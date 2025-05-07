import os.path
import shutil

import numpy as np
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt
from functools import partial

from PIL import Image

import sys


class Window(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.save_path = None

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_A:
            self.prevImage()
        if event.key() == Qt.Key_D:
            self.nextImage()
        if event.key() == Qt.Key_Delete:
            self.moveImage()
            self.nextImage()

    def initUI(self):

        screen = QDesktopWidget().screenGeometry()
        self.width, self.height = int(screen.width()), int(screen.height())
        self.top = int((screen.width() - self.width) / 2)
        self.left = int((screen.height() - self.height) / 2)

        self.label_show_iamge = QLabel(self)

        self.listView = QListWidget()
        self.listView.setContextMenuPolicy(Qt.CustomContextMenu)
        self.listView.customContextMenuRequested[QPoint].connect(self.rightMenuShow)

        self.setWindowTitle('Picture checker')
        self.setGeometry(self.top, self.left, self.width, self.height)

        layout = QVBoxLayout()
        self.open_button = QPushButton("open files")
        self.open_button.clicked.connect(self.openFolder)
        layout.addWidget(self.open_button)
        # layout.addWidget(self.listView)

        # self.prev_button = QPushButton("last")
        # self.prev_button.clicked.connect(self.prevImage)
        # self.prev_button.setDisabled(True)
        # layout.addWidget(self.prev_button)
        #
        # self.next_button = QPushButton("next")
        # self.next_button.clicked.connect(self.nextImage)
        # self.prev_button.setDisabled(True)
        # layout.addWidget(self.next_button)

        self.widget = QWidget()
        layout.addWidget(self.widget)

        self.imgName = QLabel(self)

        self.setLayout(layout)

    def openFolder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "choice img_path")
        self.save_path = os.path.join(folder_path.replace(folder_path.split('/')[-1], ""), "mask_output")
        os.makedirs(self.save_path, exist_ok=True)
        if folder_path:
            self.images = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if
                           os.path.isfile(os.path.join(folder_path, f))]
            self.images.sort()
            if self.images:
                self.current_num = 0
                self.updateImage()
                # self.updateButtons()
            else:
                self.images.clear()
                # self.updateButtons()

    def rightMenuShow(self):
        rightMenu = QMenu(self.listView)
        rightMenu.exec_(QCursor.pos())

    def updateButtons(self):
        if self.current_num < 0:
            self.prev_button.setDisabled(True)
        else:
            self.prev_button.setDisabled(False)

    def updateImage(self):
        index = self.current_num
        if index < len(self.images):
            img = Image.open(self.images[index]).convert("RGB")
            showimg = QImage(np.array(img), np.shape(img)[1], np.shape(img)[0], QImage.Format_RGB888)
            pixmap = self.loadPixmap(showimg)
            self.label_show_iamge.setPixmap(pixmap)
            self.label_show_iamge.adjustSize()
            self.label_show_iamge.setGeometry(int((self.width - np.shape(img)[1]) / 2),
                                              int((self.height - np.shape(img)[0]) / 2), np.shape(img)[1],
                                              np.shape(img)[0])
            imgName = self.images[index]
            img_num = "剩余图片数量:{}".format(str(len(self.images) - self.current_num))

            self.imgName.setText(imgName + '\n' + img_num)
            self.imgName.setAlignment(Qt.AlignTop)
            self.imgName.setIndent(2)
            self.imgName.adjustSize()
            self.imgName.setGeometry(20, 50, 1000, 50)

    def loadPixmap(self, img):
        pixmap = QPixmap.fromImage(img)
        return pixmap

    def prevImage(self):
        if self.current_num > 0:
            self.current_num -= 1
            self.updateImage()
            # self.updateButtons()

    def nextImage(self):
        if (self.current_num + 1) < len(self.images):
            self.current_num += 1
            self.updateImage()
            # self.updateButtons()

    def moveImage(self):
        if self.save_path:
            img = self.images[self.current_num]
            shutil.move(img, os.path.join(self.save_path, self.images[self.current_num].split('\\')[-1]))
            del (self.images[self.current_num])
            self.current_num -= 1
            self.images.sort()
            # self.current_num += 1


# print('1')
if __name__ == "__main__":
    app = QApplication([])
    win = Window()
    win.show()
    sys.exit(app.exec())
