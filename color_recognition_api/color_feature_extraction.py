from PIL import Image
import json
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy import stats
from collections import Counter
from skimage.color import rgb2lab, deltaE_cie76
from color_recognition_api import knn_classifier as knn_classifier

def RGB2HEX(color):
    return "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))

def color_extraction_of_test_image(test_src_image, number_of_colors, show_chart):

    # memuat gambar
    image = test_src_image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
    modified_image = cv2.resize(image, (600, 400), interpolation=cv2.INTER_AREA)
    modified_image = modified_image.reshape(modified_image.shape[0] * modified_image.shape[1], 3)
    
    clf = KMeans(n_clusters=number_of_colors, n_init=10)
    labels = clf.fit_predict(modified_image)
    
    counts = Counter(labels)
    counts = dict(sorted(counts.items()))
    
    center_colors = clf.cluster_centers_
    ordered_colors = [center_colors[i] for i in counts.keys()]
    hex_colors = [RGB2HEX(ordered_colors[i]) for i in counts.keys()]
    rgb_colors = [ordered_colors[i] for i in counts.keys()]

    red, green, blue = [], [], []

    for color in rgb_colors:
        r, g, b = color
        red.append(int(r)) 
        green.append(int(g))  
        blue.append(int(b))

    feature_data = [f"{r},{g},{b}" for r, g, b in zip(red, green, blue)]

    with open('test.data', 'w') as myfile:
        for data in feature_data:
            myfile.write(data + '\n')

    if show_chart:
        plt.figure(figsize=(8, 6))
        plt.pie(counts.values(), labels=feature_data, colors=hex_colors)
        plt.show()
        
    total_pixels = sum(counts.values())
    percentages = []
    rgb_values = []
    for i, rgb_color in enumerate(rgb_colors):
        percentage = counts[i] / total_pixels * 100
        percentages.append(percentage)
        rgb_values.append(', '.join(str(int(c)) for c in rgb_color))
        print(f"Color {i + 1}: RGB = {', '.join(str(int(c)) for c in rgb_color)}, Percentage = {percentage:.2f}%")
    return rgb_values, percentages


def color_extraction_of_training_image(img_name):

    # mendeteksi warna gambar menggunakan nama file untuk melabel data training 
    if 'red' in img_name:
        data_source = 'red'
    elif 'yellow' in img_name:
        data_source = 'yellow'
    elif 'green' in img_name:
        data_source = 'green'
    elif 'orange' in img_name:
        data_source = 'orange'
    elif 'white' in img_name:
        data_source = 'white'
    elif 'black' in img_name:
        data_source = 'black'
    elif 'blue' in img_name:
        data_source = 'blue'
    elif 'violet' in img_name:
        data_source = 'violet'
    elif 'grey' in img_name:
        data_source = 'grey'

    # memuat gambar
    image = cv2.imread(img_name)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    number_of_colors = 1
    modified_image = cv2.resize(image, (600, 400), interpolation=cv2.INTER_AREA)
    modified_image = modified_image.reshape(modified_image.shape[0] * modified_image.shape[1], 3)
    
    clf = KMeans(n_clusters=number_of_colors, n_init=10)
    labels = clf.fit_predict(modified_image)
    
    counts = Counter(labels)
    counts = dict(sorted(counts.items()))
    
    center_colors = clf.cluster_centers_
    ordered_colors = [center_colors[i] for i in counts.keys()]
    hex_colors = [RGB2HEX(ordered_colors[i]) for i in counts.keys()]
    rgb_colors = [ordered_colors[i] for i in counts.keys()]

    red, green, blue = [], [], []

    for color in rgb_colors:
        r, g, b = color
        red.append(int(r)) 
        green.append(int(g))  
        blue.append(int(b))

    feature_data = [f"{r},{g},{b}" for r, g, b in zip(red, green, blue)]

    with open('training.data', 'a') as myfile:
        for data in feature_data:
            myfile.write(data + ',' + data_source + '\n')

def training():

    # data training warna merah
    for f in os.listdir('./training_dataset/red'):
        color_extraction_of_training_image('./training_dataset/red/' + f)

    # data training warna kuning
    for f in os.listdir('./training_dataset/yellow'):
        color_extraction_of_training_image('./training_dataset/yellow/' + f)

    # data training warna hijau
    for f in os.listdir('./training_dataset/green'):
        color_extraction_of_training_image('./training_dataset/green/' + f)

    # data training warna oranye
    for f in os.listdir('./training_dataset/orange'):
        color_extraction_of_training_image('./training_dataset/orange/' + f)

    # data training warna putih
    for f in os.listdir('./training_dataset/white'):
        color_extraction_of_training_image('./training_dataset/white/' + f)

    # data training warna hitam
    for f in os.listdir('./training_dataset/black'):
        color_extraction_of_training_image('./training_dataset/black/' + f)

    # data training warna biru
    for f in os.listdir('./training_dataset/blue'):
        color_extraction_of_training_image('./training_dataset/blue/' + f)

    # data training warna abu-abu
    for f in os.listdir('./training_dataset/grey'):
        color_extraction_of_training_image('./training_dataset/grey/' + f)		


