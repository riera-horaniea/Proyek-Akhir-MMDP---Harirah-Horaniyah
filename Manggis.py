import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QPushButton, 
                             QVBoxLayout, QWidget, QFileDialog, QMessageBox, QSlider, QHBoxLayout)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt

class AdvancedMangosteenApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.title = 'Detektor Manggis'
        self.image_path = None
        self.processed_image = None
        
        # Default Threshold Values (HSV Range untuk warna cokelat/krem kelopak)
        self.h_min_val = 5   # Hue Min (Warna oranye/cokelat)
        self.s_min_val = 30  # Saturation Min
        self.v_min_val = 50  # Value Min (Kecerahan)

        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(100, 100, 900, 700) # Ukuran window lebih lebar

        # Widget Utama
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout()

        # --- Bagian Atas: Tombol & Info ---
        top_layout = QHBoxLayout()
        
        self.btn_load = QPushButton('1. Load Gambar', self)
        self.btn_load.clicked.connect(self.load_image)
        self.btn_load.setStyleSheet("padding: 10px; background-color: #2196F3; color: white; font-weight: bold;")
        top_layout.addWidget(self.btn_load)

        self.label_instruction = QLabel("Geser slider di bawah sampai HANYA kelopak yang berwarna PUTIH", self)
        self.label_instruction.setStyleSheet("font-weight: bold; color: #333;")
        top_layout.addWidget(self.label_instruction)
        
        self.main_layout.addLayout(top_layout)

        # --- Bagian Tengah: Tampilan Gambar ---
        img_layout = QHBoxLayout()
        
        # Gambar Asli (Hasil Deteksi)
        self.label_img_result = QLabel("Hasil Deteksi")
        self.label_img_result.setAlignment(Qt.AlignCenter)
        self.label_img_result.setFixedSize(400, 400)
        self.label_img_result.setStyleSheet("border: 2px solid green; background-color: #eee;")
        img_layout.addWidget(self.label_img_result)

        # Gambar Masking (Hitam Putih untuk debug)
        self.label_img_mask = QLabel("Masking (Apa yang dilihat komputer)")
        self.label_img_mask.setAlignment(Qt.AlignCenter)
        self.label_img_mask.setFixedSize(400, 400)
        self.label_img_mask.setStyleSheet("border: 2px solid black; background-color: black;")
        img_layout.addWidget(self.label_img_mask)

        self.main_layout.addLayout(img_layout)

        # --- Bagian Slider (Tuning) ---
        slider_container = QVBoxLayout()
        
        # Slider Saturation (Kekuatan Warna)
        lbl_sat = QLabel("Sensitivitas Warna (Saturation): Kurangi jika kelopak tidak terdeteksi")
        slider_container.addWidget(lbl_sat)
        self.slider_sat = QSlider(Qt.Horizontal)
        self.slider_sat.setMinimum(0)
        self.slider_sat.setMaximum(255)
        self.slider_sat.setValue(self.s_min_val)
        self.slider_sat.valueChanged.connect(self.update_processing)
        slider_container.addWidget(self.slider_sat)

        # Slider Value (Kecerahan)
        lbl_val = QLabel("Sensitivitas Cahaya (Value): Geser untuk memisahkan kelopak dari kulit")
        slider_container.addWidget(lbl_val)
        self.slider_val = QSlider(Qt.Horizontal)
        self.slider_val.setMinimum(0)
        self.slider_val.setMaximum(255)
        self.slider_val.setValue(self.v_min_val)
        self.slider_val.valueChanged.connect(self.update_processing)
        slider_container.addWidget(self.slider_val)

        self.main_layout.addLayout(slider_container)

        # --- Hasil Akhir ---
        self.result_text = QLabel("Jumlah Ruas: -", self)
        self.result_text.setAlignment(Qt.AlignCenter)
        self.result_text.setStyleSheet("font-size: 24px; font-weight: bold; color: #D32F2F; margin: 10px;")
        self.main_layout.addWidget(self.result_text)

        self.central_widget.setLayout(self.main_layout)

    def load_image(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Pilih Gambar Manggis", "", 
                                                  "Image Files (*.jpg *.jpeg *.png)", options=options)
        if file_path:
            self.image_path = file_path
            self.update_processing()

    def update_processing(self):
        if not self.image_path:
            return

        # Ambil nilai slider saat ini
        s_min = self.slider_sat.value()
        v_min = self.slider_val.value()

        # 1. Baca Gambar
        img = cv2.imread(self.image_path)
        img = cv2.resize(img, (400, 400)) # Resize agar pas di UI
        original = img.copy()
        
        # 2. Konversi ke HSV (Hue Saturation Value)
        # HSV lebih baik membedakan warna 'cokelat' kelopak dari 'ungu' kulit
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # 3. Tentukan Range Warna (Masking)
        # Target: Warna Cokelat/Oranye Kelopak
        # Hue 0-20 biasanya mencakup cokelat/oranye
        lower_brown = np.array([0, s_min, v_min]) 
        upper_brown = np.array([40, 255, 255])

        # Buat Mask (Hitam Putih)
        mask = cv2.inRange(hsv, lower_brown, upper_brown)

        # 4. Bersihkan Noise (Morphology)
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel) # Hilangkan bintik putih kecil
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel) # Tutup lubang hitam di dalam kelopak

        # Tampilkan Mask di UI Kanan
        self.display_image(mask, self.label_img_mask, is_mask=True)

        # 5. Cari Kontur pada Mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        final_count = 0
        status_msg = "Tidak ada kelopak terdeteksi"

        if contours:
            # Ambil kontur terbesar (Asumsi kelopak adalah objek terbesar yg warnanya cokelat)
            c = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(c)

            if area > 1000: # Filter jika terlalu kecil
                # Gambar kontur yang ditemukan
                cv2.drawContours(original, [c], -1, (0, 255, 0), 2)

                # Convex Hull & Defects
                hull = cv2.convexHull(c, returnPoints=False)
                try:
                    defects = cv2.convexityDefects(c, hull)
                    
                    count = 0
                    if defects is not None:
                        for i in range(defects.shape[0]):
                            s, e, f, d = defects[i, 0]
                            # d adalah kedalaman lekukan. 
                            # Kita filter lekukan yang dangkal (noise bentuk)
                            if d > 2000: 
                                count += 1
                                # Visualisasi titik lekukan (titik merah)
                                far = tuple(c[f][0])
                                cv2.circle(original, far, 5, [0, 0, 255], -1)
                    
                    # Logic: Jumlah kelopak = Jumlah lekukan (spaces between petals)
                    # Tapi kadang defect = kelopak jika bentuknya sempurna.
                    # Biasanya hasil defect count akurat untuk bentuk bintang.
                    final_count = count
                    
                    # Koreksi minimal (Manggis jarang punya < 4 kelopak)
                    if final_count < 4:
                         # Fallback logic: kadang hull menghitung sisi luar
                         # Jika gagal, kita coba pendekatan jumlah sudut ekstrem
                         status_msg = f"Terdeteksi: {final_count} (Coba geser slider)"
                    else:
                         status_msg = f"Jumlah Ruas: {final_count}"

                except Exception as e:
                    status_msg = "Error kalkulasi bentuk"
            else:
                status_msg = "Objek terlalu kecil"
        
        self.result_text.setText(status_msg)
        self.display_image(original, self.label_img_result, is_mask=False)

    def display_image(self, img, label_widget, is_mask=False):
        if is_mask:
            qformat = QImage.Format_Grayscale8
        else:
            qformat = QImage.Format_RGB888
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        h, w = img.shape[:2]
        if is_mask:
            step = w
        else:
            step = 3 * w
            
        qimg = QImage(img.data, w, h, step, qformat)
        label_widget.setPixmap(QPixmap.fromImage(qimg))

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = AdvancedMangosteenApp()
    ex.show()
    sys.exit(app.exec_())