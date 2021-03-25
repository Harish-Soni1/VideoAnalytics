from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QFileDialog, QWidget
from PyQt5.QtGui import QImage, QPixmap
import cv2
from datetime import datetime
from demo.VideoPrediction2 import *


class Ui_MainWindow(QWidget):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("QMainWindow")
        MainWindow.resize(1200, 900)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.player = QtWidgets.QLabel(self.centralwidget)
        self.player.setGeometry(QtCore.QRect(200, 20, 980, 600))
        self.player.setObjectName("Player")
        self.StopCameraButton = QtWidgets.QPushButton(self.centralwidget)
        self.StopCameraButton.setGeometry(QtCore.QRect(180, 60, 180, 60))
        self.StopCameraButton.setStyleSheet("font: 75 16pt \"Times New Roman\";")
        self.StopCameraButton.setObjectName("Stop Camera")
        self.StopCameraButton.move(10, 550)
        self.StartCameraButton = QtWidgets.QPushButton(self.centralwidget)
        self.StartCameraButton.setGeometry(QtCore.QRect(180, 60, 180, 60))
        self.StartCameraButton.setStyleSheet("font: 75 16pt \"Times New Roman\";")
        self.StartCameraButton.setObjectName("Start Camera")
        self.StartCameraButton.move(10,450)
        self.SelectImageButton = QtWidgets.QPushButton(self.centralwidget)
        self.SelectImageButton.setGeometry(QtCore.QRect(180, 60, 180, 60))
        self.SelectImageButton.setStyleSheet("font: 75 16pt \"Times New Roman\";")
        self.SelectImageButton.setObjectName("Select Image")
        self.SelectImageButton.move(10, 350)
        self.CaptureImageButton = QtWidgets.QPushButton(self.centralwidget)
        self.CaptureImageButton.setGeometry(QtCore.QRect(180, 60, 180, 60))
        self.CaptureImageButton.setStyleSheet("font: 75 16pt \"Times New Roman\";")
        self.CaptureImageButton.setObjectName("Capture Image")
        self.CaptureImageButton.move(10, 250)
        self.SelectVideoButton = QtWidgets.QPushButton(self.centralwidget)
        self.SelectVideoButton.setGeometry(QtCore.QRect(180, 60, 180, 60))
        self.SelectVideoButton.setStyleSheet("font: 75 16pt \"Times New Roman\";")
        self.SelectVideoButton.setObjectName("Select Video")
        self.SelectVideoButton.move(10, 50)
        self.PauseVideoButton = QtWidgets.QPushButton(self.centralwidget)
        self.PauseVideoButton.setGeometry(QtCore.QRect(180, 60, 180, 60))
        self.PauseVideoButton.setStyleSheet("font: 75 16pt \"Times New Roman\";")
        self.PauseVideoButton.setObjectName("Pause Video")
        self.PauseVideoButton.move(10, 150)
        self.line = QtWidgets.QFrame(self.centralwidget)
        self.line.setGeometry(QtCore.QRect(0, 640, 1200, 10))
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.line_2 = QtWidgets.QFrame(self.centralwidget)
        self.line_2.setGeometry(QtCore.QRect(500, 644, 500, 900))
        self.line_2.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_2.setObjectName("line_2")
        self.SaveDataButton = QtWidgets.QPushButton(self.centralwidget)
        self.SaveDataButton.setGeometry(QtCore.QRect(180, 60, 180, 60))
        self.SaveDataButton.setStyleSheet("font: 75 16pt \"Times New Roman\";")
        self.SaveDataButton.setObjectName("SaveDataButton")
        self.SaveDataButton.move(880, 750)
        self.Save_Data = QtWidgets.QLabel(self.centralwidget)
        self.Save_Data.setGeometry(QtCore.QRect(350, 320, 500, 1200))
        self.Save_Data.setStyleSheet("font: 8pt \"Times New Roman\";\n"
        "font: 75 16pt \"MS Shell Dlg 2\";")
        self.Save_Data.setObjectName("Save_Data")
        self.Save_Data.move(790,100)
        self.TextLabel = QtWidgets.QLabel(self.centralwidget)
        self.TextLabel.setGeometry(10, 655, 728, 234)
        self.TextLabel.setStyleSheet("font: 8pt \"Times New Roman\";\n"
        "font: 75 16pt \"MS Shell Dlg 2\";\n""border: 1px solid blacl;")
        self.TextLabel.setText("1. Here Your Dynamic Text Will Display <br> 2. Here Your Dynamic Text Will Display")
        # self.TextLabel.setText("2. Here Your Dynamic Text Will Display")
        self.BorderLabel = QtWidgets.QLabel(self.centralwidget)
        self.BorderLabel.setGeometry(762, 655, 425, 234)
        self.BorderLabel.setStyleSheet("font: 8pt \"Times New Roman\";\n"
        "font: 75 16pt \"MS Shell Dlg 2\";\n""border: 1px solid blacl;")
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        return self

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        self.capture = False
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.StartCameraButton.setText(_translate("MainWindow", "Start Camera"))
        self.SelectImageButton.setText(_translate("MainWindow", "Select Image"))
        self.SelectVideoButton.setText(_translate("MainWindow", "Select Video"))
        self.StopCameraButton.setText(_translate("MainWindow", "Stop Camera"))
        self.PauseVideoButton.setText(_translate("MainWindow", "Pause Video"))
        self.CaptureImageButton.setText(_translate("MainWindow", "Capture"))
        self.SaveDataButton.setText(_translate("MainWindow", "Save Data"))
        self.Save_Data.setText(_translate("MainWindow", "Save Data In Excel Sheet"))
        self.StartCameraButton.clicked.connect(self.startCamera)
        self.StopCameraButton.clicked.connect(self.stopCamera)
        self.SelectImageButton.clicked.connect(self.uploadHandlerForImage)
        self.SelectVideoButton.clicked.connect(self.uploadHandlerForVideo)
        self.PauseVideoButton.clicked.connect(self.pauseVideo)
        self.CaptureImageButton.clicked.connect(self.captureImage)

    def uploadHandlerForImage(self):
        path = self.openDialogBoxForImage()
        image = path[0]
        self.displayImage(cv2.imread(image))

    def uploadHandlerForVideo(self):
        self.ret = True
        self.player.setEnabled(True)
        try:
            path = self.openDialogBoxForVideo()
            video = path[0]
            prediction_on_video(video)
            #cap = cv2.VideoCapture(video)
            # while (cap.isOpened()):
            #     ret, frame = cap.read()
            #     if frame is not None:
            #         if self.ret:
            #             if ret:
            #                 self.displayImage(frame)
            #                 cv2.waitKey()
            #                 if (self.capture == True):
            #                     self.name = "Capture_Image_" + str(int(datetime.timestamp(datetime.now())))
            #                     cv2.imwrite(
            #                         "Capture Images/%s.png" % (self.name), frame)
            #                     self.capture = False
            #             else:
            #                 cap.release()
            #                 cv2.destroyAllWindows()
            #                 break
            #         else:
            #             cap.release()
            #             self.player.clear()
            #             cv2.destroyAllWindows()
            #             break
            #     else:
            #         self.player.clear()
            #         break
            #
            # cap.release()
            # cv2.destroyAllWindows()
            #self.player.clear()

        except Exception as e:
            print(str(e))

    def displayImage(self, img):
        try:
            qformat = QImage.Format_Indexed8
            if len(img.shape) == 3:
                if (img.shape[2]) == 4:
                    qformat = QImage.Format_RGB888
                else:
                    qformat = QImage.Format_RGB888

            img = QImage(img, img.shape[1], img.shape[0], qformat)
            img = img.rgbSwapped()
            self.player.setScaledContents(True)
            self.player.setPixmap(QPixmap.fromImage(img))

        except Exception as e:
            print(str(e))

    def openDialogBoxForImage(self):
        file = QFileDialog.getOpenFileName(self, 'Open file', '', "Image files (*.jpg *.jpeg *.png)")
        return file

    def openDialogBoxForVideo(self):
        file = QFileDialog.getOpenFileName(self, 'Open file', '', "Video files (*.mp4)")
        return file

    def startCamera(self):
        self.ret = True
        self.player.setEnabled(True)
        cap = cv2.VideoCapture(0)
        while (cap.isOpened()):
            ret, frame = cap.read()
            if frame is not None:
                if self.ret == True:
                    self.displayImage(frame)
                    cv2.waitKey()
                    if (self.capture == True):
                        self.name = "Capture_Image_" + str(int(datetime.timestamp(datetime.now())))
                        cv2.imwrite(
                            "Capture Images/%s.png" % (self.name), frame)
                        self.capture = False
                else:
                    cap.release()
                    cv2.destroyAllWindows()

            else:
                self.player.clear()

        cap.release()
        cv2.destroyAllWindows()

    def stopCamera(self):
        self.ret = False
        self.player.clear()
        return self

    def pauseVideo(self):
        self.ret = False
        self.player.clear()
        return self

    def captureImage(self):
        self.capture=True


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    form = QtWidgets.QWidget()
    ui = Ui_MainWindow()
    ui.setupUi(form)
    form.show()
    sys.exit(app.exec_())
