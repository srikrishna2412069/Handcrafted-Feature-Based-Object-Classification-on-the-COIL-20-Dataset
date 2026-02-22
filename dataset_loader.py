
import os
import cv2
import numpy as np

def load_dataset(dataset_path):
    images = []
    labels = []

    for filename in os.listdir(dataset_path):
        if filename.endswith('.png'):
            img_path = os.path.join(dataset_path, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = img / 255.0
            images.append(img)

            label = int(filename.split('__')[0].replace('obj',''))
            labels.append(label)

    X = np.array(images)
    y = np.array(labels)

    print('Dataset shape:', X.shape)
    return X, y
