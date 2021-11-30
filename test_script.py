import librosa
import glob
import logging
import numpy as np
import cv2
import random
import pandas as pd
from matplotlib import pyplot as plt
from tensorflow.keras.models import model_from_json


def file_load(wav_name, mono=False):
    try:
        return librosa.load(wav_name, sr=None, mono=mono)
    except:
        logging.error("file broken or not exists!:{}".format(wav_name))


def make_data(folder_name, id_name):
    result = []
    all_name = glob.glob(folder_name)
    for name in all_name:
        if id_name in name:
            # append ndarray
            result.append(file_load(name)[0])
    return np.array(result)


# change wav data to STFT
def to_sp(x):
    stft = librosa.stft(x, n_fft=256, hop_length=256)
    sp = librosa.amplitude_to_db(np.abs(stft))
    return sp


def to_img(x):
    result = []
    for i in range(len(x)):
        result.append(cv2.resize(to_sp(x[i]), (224, 224)))
    return np.array(result)


machine = 'slider'
id_no = 'id_00'

x_train = make_data("./data/" + machine + "/train/*", "normal_" + id_no)
x_test_normal = make_data("./data/" + machine + "/test/*", "normal_" + id_no)
x_test_anomaly = make_data("./data/" + machine + "/test/*", "anomaly_" + id_no)


# max_ = np.max(x_train)
# min_ = np.min(x_train)
#
# print("max:", max_)
# # 0.18673706
# print(("min:", min_))
# # -0.18011475
# X_train = (to_img(x_train) - min_) / (max_ - min_)
# X_test_normal = (to_img(x_test_normal) - min_) / (max_ - min_)
# X_test_anomaly = (to_img(x_test_anomaly) - min_) / (max_ - min_)
#
# mean_ = np.mean(X_train)
# sigma = np.std(X_train)
#
# print("mean", mean_)
# # -87.50292
# print("sigma", sigma)
# # 26.944086
#
# X_train = (X_train * 255 - mean_) / sigma
# X_test_normal = (X_test_normal * 255 - mean_) / sigma
# X_test_anomaly = (X_test_anomaly * 255 - mean_) / sigma
#
# X_train = np.expand_dims(X_train, axis=-1)
# X_test_normal = np.expand_dims(X_test_normal, axis=-1)
# X_test_anomaly = np.expand_dims(X_test_anomaly, axis=-1)


def normalization(x):
    max_ = 0.18673706  # np.max(x_train)
    min_ = -0.18011475  # np.min(x_train)
    mean = -87.50292
    sigma = 26.944086

    result = cv2.resize(x, (224, 224))
    result = (result - min_) / (max_ - min_)
    return (result * 255 - mean) / sigma


def to_img2(x):
    result = cv2.resize(to_sp(x), (224, 224))
    return np.array(result)


def add_whitenoise(x, rate=0.002):
    return to_sp(x + rate * np.random.randn(len(x)))


def add_pinknoise(x, ncols=11, alpha=0.002):
    """Generates pink noise using the Voss-McCartney algorithm.

    nrows: number of values to generate
    rcols: number of random sources to add

    returns: NumPy array
    """
    nrows = len(x)
    array = np.empty((nrows, ncols))
    array.fill(np.nan)
    array[0, :] = np.random.random(ncols)
    array[:, 0] = np.random.random(nrows)

    # the total number of changes is nrows
    n = nrows
    cols = np.random.geometric(0.5, n)
    cols[cols >= ncols] = 0
    rows = np.random.randint(nrows, size=n)
    array[rows, cols] = np.random.random(n)

    df = pd.DataFrame(array)
    df.fillna(method='ffill', axis=0, inplace=True)
    total = df.sum(axis=1)

    return to_sp(alpha * total.values + x)


def draw_line(x, length=[5, 20], thickness_length=[2, 4]):
    result = np.copy(x)
    width = x.shape[1]
    height = x.shape[0]
    angle = [0, np.pi / 2, np.pi, np.pi * 3 / 2]
    np.random.shuffle(angle)

    length = np.random.randint(length[0], length[1])
    x1 = random.randint(length, width - length)
    x2 = x1 + length * np.cos(angle[0])
    y1 = random.randint(length, height - length)
    y2 = y1 + length * np.sin(angle[0])

    thickness = random.randint(thickness_length[0], thickness_length[1])
    color1 = float(np.max(x))

    cv2.line(result, (x1, y1), (int(np.min([width, x2])), int(np.min([height, y2]))), color1, thickness)

    return result


def average_Hz(x, length=[2, 4]):
    result = np.copy(x)
    height = x.shape[0]

    length = np.random.randint(length[0], length[1])
    begin = np.random.randint(0, height - length)
    for i in range(length):
        result[begin + i] = np.mean(result[begin + i])

    return result


def show_plot():
    target = np.copy(x_train[0])

    img0 = to_img2(target)
    img0 = normalization(img0)

    img1 = draw_line(img0)

    img2 = add_whitenoise(target)
    img2 = normalization(img2)

    img3 = add_pinknoise(target)
    img3 = normalization(img3)

    img4 = average_Hz(img0)

    plt.figure(figsize=(20, 20))
    plt.subplot(1, 5, 1)
    plt.axis("off")
    plt.title("normal data")
    plt.imshow(img0)

    plt.subplot(1, 5, 2)
    plt.axis("off")
    plt.title("slight line")
    plt.imshow(img1)

    plt.subplot(1, 5, 3)
    plt.axis("off")
    plt.title("add white noise")
    plt.imshow(img2)

    plt.subplot(1, 5, 4)
    plt.axis("off")
    plt.title("add pink noise")
    plt.imshow(img3)

    plt.subplot(1, 5, 5)
    plt.axis("off")
    plt.title("average Hz")
    plt.imshow(img4)
    plt.show()


