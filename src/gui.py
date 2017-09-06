import sys
from os.path import expanduser

from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog
from PyQt5 import uic, QtGui
import src.resnet_main

Ui_MainWindow, QtBaseClass = uic.loadUiType("mainGui_3A.ui")


class MyApp(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MyApp, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # set logo of app and fixing the screen size
        logo_src = "appLogo.png"
        self.setWindowIcon(QtGui.QIcon(logo_src))
        self.setFixedSize(self.size())

        # initializing variables and setting event listeners
        self.ui.pushButton_inputDirectory.clicked.connect(self.choose_input_directory)
        self.ui.pushButton_outputDirectory.clicked.connect(self.choose_output_directory)
        self.ui.pushButton_CreateAlbum.clicked.connect(self.create_album)
        self.ui.pushButton_inputDirectory.setToolTip('Choose input directory')
        self.ui.pushButton_outputDirectory.setToolTip('Choose output directory')


    def choose_input_directory(self):
        input_dir = QFileDialog.getExistingDirectory(None, 'Select a folder:', expanduser("~"))
        self.ui.lineEdit_inputDirectory.setText(input_dir)

    def choose_output_directory(self):
        input_dir = QFileDialog.getExistingDirectory(None, 'Select a folder:', expanduser("~"))
        self.ui.lineEdit_outputDirectory.setText(input_dir)

    def create_album(self):
        input_dir = self.ui.lineEdit_inputDirectory.text()
        output_dir = self.ui.lineEdit_outputDirectory.text()
        album_name = self.ui.lineEditAlbumName.text()
        if input_dir != "" and output_dir != "":
            self.ui.status.setText("Status: Creating Album '" + album_name + "' ...")
            if (self.ui.radioButton_manually.isChecked()):
                images_number = self.ui.spinBox_image_number.value()
                src.resnet_main.create_album(album_name, input_dir, output_dir, images_number)
            else:
                src.resnet_main.create_album(album_name, input_dir, output_dir,None)
            self.ui.status.setText("Status: Album '" + album_name + "' was created!")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MyApp()
    window.show()
    sys.exit(app.exec_())
