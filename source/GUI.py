from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from color_recognition_api import color_feature_extraction
from color_recognition_api import knn_classifier
from collections import Counter
from skimage.color import rgb2lab, deltaE_cie76
import os

class ColorClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Color Classifier")
        self.root.geometry("1024x768")  # Menambahkan ukuran frame 200x200

        # Tombol untuk memilih gambar
        self.choose_image_button = tk.Button(root, text="Choose Image", command=self.choose_image)
        self.choose_image_button.place(x=50, y=20)

        # Tombol untuk otsu segmentation
        self.otsu_segmentation_button = tk.Button(root, text="Segmentation", command=self.show_segment)
        self.otsu_segmentation_button.place(x=435, y=20)

        # Tombol untuk ekstraksi warna
        self.color_extraction_button = tk.Button(root, text="Color Extraction", command=self.color_extraction)
        self.color_extraction_button.place(x=750, y=20)

        # Tombol untuk prediksi
        self.predict_color_button = tk.Button(root, text="Predict", command=self.predict_color)
        self.predict_color_button.place(x=900, y=20)

        # Area untuk menampilkan gambar
        self.image_label = tk.Label(root)
        self.image_label.place(x=20, y=70)

        # Area untuk menampilkan gambar segmentasi
        self.segmentation_image_label = tk.Label(root)
        self.segmentation_image_label.place(x=250, y=70)
        self.segmentation_label = tk.Label(root, text="Hasil Segmentasi", font=("Arial", 10))  
        self.segmentation_label.place(x=300, y=285)

        # Area untuk menampilkan gambar Dilasi
        self.dilated_image_label = tk.Label(root)
        self.dilated_image_label.place(x=500, y=70)
        self.dilated_label = tk.Label(root, text="Hasil Dilasi", font=("Arial", 10))  
        self.dilated_label.place(x=565, y=285)

        # Area untuk menampilkan gambar Erosi
        self.eroded_image_label = tk.Label(root)
        self.eroded_image_label.place(x=250, y=310)
        self.eroded_label = tk.Label(root, text="Hasil Erosi", font=("Arial", 10))  
        self.eroded_label.place(x=315, y=530)

        # Area untuk menampilkan gambar bitwise_and
        self.bitwise_image_label = tk.Label(root)
        self.bitwise_image_label.place(x=500, y=310)
        self.bitwise_label = tk.Label(root, text="Hasil Bitwise_And", font=("Arial", 10))  
        self.bitwise_label.place(x=545, y=530)

        # Variabel untuk menyimpan hasil ekstraksi warna
        self.extraction_label = tk.Label(root, text="Extracted RGB: ", font=("Arial", 12))  
        self.extraction_label.place(x=750, y=70)
        
        # Variabel untuk menyimpan hasil prediksi
        self.prediction_label = tk.Label(root, text="Detected color: ", font=("Arial", 12))  
        self.prediction_label.place(x=10, y=600)

        # Variabel untuk menyimpan gambar yang dipilih
        self.source_image = None
        
        # Variabel untuk menyimpan persentase warna
        self.extracted_percentage = None
        
    # pilih gambar
    def choose_image(self):
        # Menggunakan file dialog untuk memilih gambar
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])

        if file_path:
            # Membaca gambar
            self.source_image = cv2.imread(file_path)
            self.display_image(self.source_image)

    # segmentasi
    def otsu_segmentation(self, image):
        # Mengonversi gambar ke skala abu-abu
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Melakukan Otsu Thresholding
        _, segmented_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        return segmented_image

    # dilasi
    def perform_dilation(self, segmented_image):
        # Dilasi
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        dilated_image = cv2.dilate(segmented_image, kernel, iterations=1)

        return dilated_image

    # erosi
    def perform_erosion(self, dilated_image):
        # Erosi
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        eroded_image = cv2.erode(dilated_image, kernel, iterations=1)

        return eroded_image

    # bitwise
    def perform_bitwise_and(self, image, eroded_image):
        # Menggabungkan hasil segmentasi dengan gambar asli
        segmented_image = cv2.cvtColor(eroded_image, cv2.COLOR_GRAY2BGR)
        bitwise_image = cv2.bitwise_and(image, segmented_image)

        return bitwise_image

    # ekstrak warna
    def color_extraction(self):
        # Memastikan data training sudah siap
        PATH_TRAIN = './training.data'
        PATH_TEST = './test.data'
        if os.path.isfile(PATH_TRAIN) and os.access(PATH_TRAIN, os.R_OK) and os.path.isfile(PATH_TEST) and os.access(PATH_TEST, os.R_OK):
            print('Training data is ready, classifier is loading...')
        else:
            print('Training data is being created...')
            open('training.data', 'w')
            open('test.data', 'w')
            color_feature_extraction.training()
            print('Training data is ready, classifier is loading...')

        # Mengekstrak fitur
        if self.source_image is not None:
            segmented_image = self.otsu_segmentation(self.source_image)
            dilated_image = self.perform_dilation(segmented_image)
            eroded_image = self.perform_erosion(dilated_image)
            bitwise_image = self.perform_bitwise_and(self.source_image, eroded_image)
            hex_color, rgb_values, extracted_color_percentages = color_feature_extraction.color_extraction_of_test_image(bitwise_image, 8)

            extraction_text = "Extracted RGB: \n"
            # Membuat list dari nilai RGB yang diekstraksi
            extracted_values = [f"RGB = {rgb}" for rgb in rgb_values]
            self.extracted_percentage = extracted_color_percentages
            extraction_text += "\n".join(extracted_values)
            self.extraction_label.config(text=extraction_text)

            plt.figure(figsize=(8, 6))
            plt.pie(extracted_color_percentages, labels=rgb_values, colors=hex_color)
            plt.show()
        else:
            print("No image selected!")

    # prediksi
    def predict_color(self):
        # Memastikan data training sudah siap
        extracted_percentage = self.extracted_percentage
        PATH_TRAIN = './training.data'
        PATH_TEST = './test.data'
        if os.path.isfile(PATH_TRAIN) and os.access(PATH_TRAIN, os.R_OK) and os.path.isfile(PATH_TEST) and os.access(PATH_TEST, os.R_OK):
            if self.source_image is not None:
                predictions = knn_classifier.main('training.data', 'test.data')
                prediction_text = "Detected colors = "
                for i in range(len(predictions)):
                    prediction_text += f"{predictions[i]} ({extracted_percentage[i]:.2f}%)"
                    if i < len(predictions) - 1:
                        prediction_text += ", "
                self.prediction_label.config(text=prediction_text)
            else:
                print("No image selected!")
        else:
            print("No Extracted RGB!")         


    def display_image(self, image):
        # Konversi gambar OpenCV ke format yang dapat ditampilkan oleh Tkinter
        image = cv2.resize(image, (200, 200), interpolation=cv2.INTER_AREA)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = ImageTk.PhotoImage(image)

        # Menampilkan gambar di GUI
        self.image_label.config(image=image)
        self.image_label.image = image

    def show_segment(self):
        if self.source_image is not None:
            segmented_image = self.otsu_segmentation(self.source_image)
            dilated_image = self.perform_dilation(segmented_image)
            eroded_image = self.perform_erosion(dilated_image)
            final_image = self.perform_bitwise_and(self.source_image, eroded_image)
            
            # Display all processed images
            self.display_segmented_image(segmented_image, dilated_image, eroded_image, final_image)
        else:
            print("No image selected for segmentation!")

    def display_segmented_image(self, segmented_image, dilated_image, eroded_image, final_image):
        # Konversi gambar hasil segmentasi OpenCV ke format yang dapat ditampilkan oleh Tkinter
        segmented_image_display = cv2.resize(segmented_image, (200, 200), interpolation=cv2.INTER_AREA)
        segmented_image_display = cv2.cvtColor(segmented_image_display, cv2.COLOR_BGR2RGB)
        segmented_image_display = Image.fromarray(segmented_image_display)
        segmented_image_display = ImageTk.PhotoImage(segmented_image_display)  

        # Konversi gambar hasil dilasi OpenCV ke format yang dapat ditampilkan oleh Tkinter
        dilated_image_display = cv2.resize(dilated_image, (200, 200), interpolation=cv2.INTER_AREA)
        dilated_image_display = cv2.cvtColor(dilated_image_display, cv2.COLOR_BGR2RGB)
        dilated_image_display = Image.fromarray(dilated_image_display)
        dilated_image_display = ImageTk.PhotoImage(dilated_image_display)
        
        # Konversi gambar hasil erosi OpenCV ke format yang dapat ditampilkan oleh Tkinter
        eroded_image_display = cv2.resize(eroded_image, (200, 200), interpolation=cv2.INTER_AREA)
        eroded_image_display = cv2.cvtColor(eroded_image_display, cv2.COLOR_BGR2RGB)
        eroded_image_display = Image.fromarray(eroded_image_display)
        eroded_image_display = ImageTk.PhotoImage(eroded_image_display)
        
        # Konversi gambar hasil bitwise operation OpenCV ke format yang dapat ditampilkan oleh Tkinter
        final_image_display = cv2.resize(final_image, (200, 200), interpolation=cv2.INTER_AREA)
        final_image_display = cv2.cvtColor(final_image_display, cv2.COLOR_BGR2RGB)
        final_image_display = Image.fromarray(final_image_display)
        final_image_display = ImageTk.PhotoImage(final_image_display)
        
        # Menampilkan gambar hasil segmentasi di GUI
        self.segmentation_image_label.config(image=segmented_image_display)  
        self.dilated_image_label.config(image=dilated_image_display)
        self.eroded_image_label.config(image=eroded_image_display)
        self.bitwise_image_label.config(image=final_image_display)

        # Keeping references to the images
        self.segmentation_image_label.image = segmented_image_display
        self.dilated_image_label.image = dilated_image_display
        self.eroded_image_label.image = eroded_image_display
        self.bitwise_image_label.image = final_image_display



if __name__ == "__main__":
    # Membuat instance Tkinter
    root = tk.Tk()
    app = ColorClassifierApp(root)

    # Menjalankan GUI
    root.mainloop()
