import sys, os, time
from PyQt5.QtWidgets import QDialog, QLineEdit, QComboBox, QApplication, QMainWindow, QLabel, QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog, QWidget, QRubberBand
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt, QRect, QPoint, QSize

import DataBase.database as db
from Compare import compare as cp
from Compare import sift


class GUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        
        self.database_path="C:/Users/asus/GitHub_clones/DIP23/src/resources/data_base"
        self.setWindowTitle('Image Processing App')
        self.setGeometry(100, 100, 800, 600)

        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)

        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)

        self.buttons_layout = QVBoxLayout()

        self.open_image_button = QPushButton('Open Image', self)
        self.open_image_button.clicked.connect(self.open_image)
        self.open_image_button.setFixedSize(150, 30)

        self.compute_button = QPushButton('Compute', self)
        self.compute_button.clicked.connect(self.compute)
        self.compute_button.setFixedSize(150, 30)

        self.options_combo_box = QComboBox(self)
        self.options_combo_box.addItems(['SIFT', 'Alessandro', 'Jakub' ])
        self.options_combo_box.currentIndexChanged.connect(self.update_compute_function)
        self.options_combo_box.setFixedSize(150, 30)
        
        self.lime_button = QPushButton('LIME', self)
        self.lime_button.clicked.connect(self.computeLIME)
        self.lime_button.setFixedSize(150, 30)
        
        self.predicted_value_label = QLabel('Predicted Flower:', self)
        self.confidence_label = QLabel('Confidence:', self)
        self.status_label = QLabel('Working...', self)
        self.predicted_value_label.setFixedSize(150, 30)
        self.confidence_label.setFixedSize(150, 30)
        self.status_label.setFixedSize(150, 30)

        self.predicted_value_textbox = QLineEdit(self)
        self.confidence_textbox = QLineEdit(self)
        self.predicted_value_textbox.setFixedSize(150, 30)
        self.confidence_textbox.setFixedSize(150, 30)

        self.predicted_value_textbox.setReadOnly(True)
        self.confidence_textbox.setReadOnly(True)

        self.predicted_value_label.setBuddy(self.predicted_value_textbox)
        self.confidence_label.setBuddy(self.confidence_textbox)

        self.buttons_layout.addWidget(self.open_image_button)
        self.buttons_layout.addWidget(self.options_combo_box)
        self.buttons_layout.addWidget(self.compute_button)
        self.buttons_layout.addWidget(self.lime_button)
        self.buttons_layout.addWidget(self.predicted_value_label)
        self.buttons_layout.addWidget(self.predicted_value_textbox)
        self.buttons_layout.addWidget(self.confidence_label)
        self.buttons_layout.addWidget(self.confidence_textbox)
        self.buttons_layout.addWidget(self.status_label)
        self.buttons_layout.addStretch()
        
        self.status_label.hide()

        self.central_layout = QHBoxLayout()
        self.central_layout.addLayout(self.buttons_layout)
        self.central_layout.addWidget(self.image_label)

        self.central_widget.setLayout(self.central_layout)

        self.image_path = None  # To store the current image path
        self.selection_start = QPoint()
        self.selection_end = QPoint()
        self.rect_size = 0
        self.rubber_band = QRubberBand(QRubberBand.Rectangle, self.image_label)
        self.selected_option = 'SIFT'
        self.temp_path = os.path.dirname(os.path.abspath(__file__)) + "/temp.jpg"
        
        self.database = db.Database(self.database_path)
        self.image_comparer = cp.Compare(sift.Sift())

    def open_image(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_dialog = QFileDialog()
        self.image_path, _ = file_dialog.getOpenFileName(self, "Open Image File", "", "Images (*.png *.jpg *.bmp *.gif *.tif);;All Files (*)", options=options)

        if self.image_path:
            pixmap = QPixmap(self.image_path)
            self.image_label.setPixmap(pixmap)
            self.image_label.adjustSize()
            
            # Save the image to the same folder with the name "temp.jpg"
            pixmap.save(self.temp_path, 'JPG')

    def compute(self):
        if self.selected_option == 'Alessandro':
            self.computeAlessandro()
        elif self.selected_option == 'Jakub':
            self.computeJakub()
        elif self.selected_option == 'SIFT':
            self.computeRenato()

    def computeAlessandro(self):        
        print("Not computing.")
        return
        
    def computeLIME(self):
        self.status_label.show()
        time.sleep(1)
        self.model.saliency(self.temp_path)
        self.status_label.hide()
        self.show_image("/heatmap.png")

    def computeJakub(self):
        print("Not computing.")

    def computeRenato(self):
        print("Compute Renato is called.")
        prediction = self.image_comparer.comparation(self.database, self.temp_path)
        self.update_results(prediction.folder_name, str(prediction.similarity_score))
        self.status_label.hide()
        
        
    def update_results(self, predicted_value, confidence):
        self.predicted_value_textbox.setText(predicted_value)
        self.confidence_textbox.setText(confidence)
        
    def mousePressEvent(self, event):
        if self.rubber_band.isVisible():
            self.rubber_band.clearMask()
            self.rubber_band.hide()
        if event.button() == Qt.LeftButton and self.image_path:
            self.selection_start = self.image_label.mapFromGlobal(event.globalPos())
            self.rubber_band.setGeometry(QRect(self.selection_start, QSize()))
            self.rubber_band.show()

    def mouseMoveEvent(self, event):
        if self.rubber_band.isVisible():
            self.selection_end = self.image_label.mapFromGlobal(event.globalPos())
            self.rect_size = min(abs(self.selection_end.x()-self.selection_start.x()), 
                                   abs(self.selection_end.y()-self.selection_start.y()))
            self.rubber_band.setGeometry(QRect(self.selection_start, QSize(self.rect_size, self.rect_size)))

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.rubber_band.isVisible():
            self.selection_end = self.image_label.mapFromGlobal(event.globalPos())
            self.rect_size = min(abs(self.selection_end.x()-self.selection_start.x()), 
                                   abs(self.selection_end.y()-self.selection_start.y()))
            self.rubber_band.setGeometry(QRect(self.selection_start, QSize(self.rect_size, self.rect_size)))

    def update_compute_function(self, index):
        options = ['Alessandro', 'Jakub', 'Renato']
        self.selected_option = options[index]    
        
        # Show or hide LIME button based on the selected option
        if self.selected_option == 'Alessandro':
           self.lime_button.show()
        else:
           self.lime_button.hide()
           
    def show_image(self, image_name):
        dialog = QDialog(self)
        dialog.setWindowTitle('LIME Heatmap')
        dialog.setGeometry(150, 150, 200, 200)

        layout = QVBoxLayout()
        heatmap_path = os.path.dirname(os.path.abspath(__file__)) + str(image_name)
        
        image_label = QLabel(dialog)
        pixmap = QPixmap(heatmap_path)
        image_label.setPixmap(pixmap)
        image_label.setAlignment(Qt.AlignCenter)

        layout.addWidget(image_label)

        dialog.setLayout(layout)
        dialog.exec_()
            
            
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = GUI()
    window.show()
    sys.exit(app.exec_())