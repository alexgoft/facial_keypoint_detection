import pandas as pd
import numpy as np
# import tensorflow as tf
# import cv2

from os.path import exists

SIGMA = 0.8
X_TRAIN_PATH = 'Data/X_train.npy'
Y_TRAIN_PATH = 'Data/Y_train.npy'
H = 96
W = 96

"""
    left_eye_center_x
    left_eye_center_y
    right_eye_center_x
    right_eye_center_y
    left_eye_inner_corner_x
    left_eye_inner_corner_y
    left_eye_outer_corner_x
    left_eye_outer_corner_y
    right_eye_inner_corner_x
    right_eye_inner_corner_y
    right_eye_outer_corner_x
    right_eye_outer_corner_y
    left_eyebrow_inner_end_x
    left_eyebrow_inner_end_y
    left_eyebrow_outer_end_x
    left_eyebrow_outer_end_y
    right_eyebrow_inner_end_x
    right_eyebrow_inner_end_y
    right_eyebrow_outer_end_x
    right_eyebrow_outer_end_y
    nose_tip_x
    nose_tip_y
    mouth_left_corner_x
    mouth_left_corner_y
    mouth_right_corner_x
    mouth_right_corner_y
    mouth_center_top_lip_x
    mouth_center_top_lip_y
    mouth_center_bottom_lip_x
    mouth_center_bottom_lip_y
    Image
"""


def point_gaussian_value(p1, p2, sigma=0.8):
    x1, y1 = p1
    x2, y2 = p2

    return np.exp(-1 * (np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) / sigma ** 2))


def prepare_data():
    
    # Read train data
    df = pd.read_csv('Data/training.zip', compression='zip', header=0, sep=',', quotechar='"')
    header = list(df.columns.values)

    # Create train data
    if exists(X_TRAIN_PATH) and exists(Y_TRAIN_PATH):
        X_train = np.load(X_TRAIN_PATH)
        Y_train = np.load(Y_TRAIN_PATH)
    else:
        X_train = []
        Y_train = []

        rows_idx = np.arange(H)
        cols_idx = np.arange(W)

        for index, row in df.iterrows():

            print(index)

            sample = np.reshape(a=np.array(row['Image'].split(' '), dtype=np.uint8), newshape=(H, W))
            sample_annotation = []

            for i in range(0, len(header) - 1, 2):
                x, y = (row[header[i]], row[header[i + 1]])
                point_annotation = np.zeros(shape=(H, W)) \
                    if np.isnan(x) or np.isnan(y) \
                    else np.exp(-1 * (np.sqrt((x - rows_idx[:, None]) ** 2 + (y - cols_idx) ** 2) / SIGMA ** 2))

                sample_annotation.append(point_annotation)

            X_train.append(sample)
            Y_train.append(sample_annotation)

        X_train = np.array(X_train)
        Y_train = np.array(Y_train)

        np.save(X_TRAIN_PATH, X_train)
        np.save(Y_TRAIN_PATH, Y_train)

    print(X_train.shape)
    print(Y_train.shape)

    return X_train, Y_train


def prepare_model():
    return None


def main():

    X_train, Y_train = prepare_data()
    model = prepare_model()



if __name__ == '__main__':
    main()
