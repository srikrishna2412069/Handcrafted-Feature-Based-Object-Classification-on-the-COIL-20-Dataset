
import numpy as np
from skimage.feature import hog

def extract_hog_features(X):
    hog_features = []

    for img in X:
        features = hog(
            img,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            block_norm='L2-Hys'
        )
        hog_features.append(features)

    X_features = np.array(hog_features)
    print('HOG feature shape:', X_features.shape)
    return X_features
