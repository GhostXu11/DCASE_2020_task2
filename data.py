import sys
import librosa
import numpy
import numpy as np
import os
import scipy.io.wavfile as wav
import torch
from python_speech_features import *


def one_dimentional_data(data_dir='../../data/xingqing/data_dcase2020_task2/data/', type='fan', ID='id_00',
                         winlen=0.064, winstep=0.032, numcep=128, ratio_train=1):
    frequency_spectrum_train = []
    frequency_spectrum_test_nor = []
    frequency_spectrum_test_anor = []
    data_dir = data_dir + type
    files1 = os.listdir(data_dir + './train')
    files2 = os.listdir(data_dir + './test')

    i = 0
    j = 0
    k = 0
    for item in files1:
        if ID in item:
            samplerate, train_signal = wav.read(data_dir + './train/' + item)
            train_freq = mfcc(train_signal, samplerate, winlen=winlen, winstep=winstep, numcep=numcep, nfilt=numcep,
                              lowfreq=0, highfreq=None, preemph=0.97, ceplifter=22, appendEnergy=False,
                              winfunc=numpy.hamming)

            if 'normal' in item:
                label = 1
            else:
                label = 2

            if i == 0:
                frequency_spectrum_train = train_freq
            else:
                frequency_spectrum_train = np.concatenate((frequency_spectrum_train, train_freq), axis=0)

            i += 1

    for item in files2:
        if ID in item:
            samplerate, test_signal = wav.read(data_dir + '/test/' + item)
            test_frqe = mfcc(test_signal, samplerate, winlen=winlen, winstep=winstep, numcep=numcep,
                             nfilt=numcep, nfft=int(16000 * winlen), lowfreq=0, highfreq=None, preemph=0.97,
                             ceplifter=22, appendEnergy=True,
                             winfunc=numpy.hamming)

            if 'normal' in item:
                label = 1
                if j == 0:
                    frequency_spectrum_test_nor = test_frqe
                else:
                    frequency_spectrum_test_nor = np.concatenate((frequency_spectrum_test_nor, test_frqe),
                                                                 axis=0)
                j += 1
            else:
                label = 2
                if k == 0:
                    frequency_spectrum_test_anor = test_frqe
                else:
                    frequency_spectrum_test_anor = np.concatenate((frequency_spectrum_test_anor, test_frqe),
                                                                  axis=0)
                k += 1
    data1 = np.reshape(frequency_spectrum_train, (-1, 8, 128))
    data2 = np.reshape(frequency_spectrum_test_nor, (-1, 8, 128))
    data3 = np.reshape(frequency_spectrum_test_anor, (-1, 8, 128))
    return data1, data1, data2, data3


def file_to_vector_array(y, sr, n_mels=128, frames=5, n_fft=1024, hop_length=512, power=2.0):
    """
    convert file_name to a vector array.
    file_name : str
        target .wav file
    return : numpy.array( numpy.array( float ) )
        vector array
        * dataset.shape = (dataset_size, feature_vector_length)
    """
    # 01 calculate the number of dimensions
    dims = n_mels * frames

    # 02 generate melspectrogram using librosa
    mel_spectrogram = librosa.feature.melspectrogram(y=y,
                                                     sr=sr,
                                                     n_fft=n_fft,
                                                     hop_length=hop_length,
                                                     n_mels=n_mels,
                                                     power=power)  # [128,313]

    # 03 convert melspectrogram to log mel energy
    log_mel_spectrogram = 20.0 / power * numpy.log10(mel_spectrogram + sys.float_info.epsilon)

    # 04 calculate total vector size
    vector_array_size = len(log_mel_spectrogram[0, :]) - frames + 1

    # 05 skip too short clips
    if vector_array_size < 1:
        return numpy.empty((0, dims))

    # 06 generate feature vectors by concatenating multiframes
    vector_array = numpy.zeros((vector_array_size, dims))
    for t in range(frames):
        vector_array[:, n_mels * t: n_mels * (t + 1)] = log_mel_spectrogram[:, t: t + vector_array_size].T

    return vector_array  # [309,640]


def dnn_data_train(datadir='../../../data/xingqing/data_dcase2020_task2/data/', type='fan', ID='id_00', frames=5,
                   numcep=128):
    frequency_spectrum_train = []
    frequency_spectrum_test = []
    datadir = datadir + type
    files1 = os.listdir(datadir + '/train')
    # files2 = os.listdir(datadir + '/test')
    i = 0
    j = 0
    k = 0
    for everyone in files1:
        if ID in everyone:
            # print(everyone)
            train_signal, samplerate = librosa.load(datadir + '/train/' + everyone, sr=None, mono=False)
            train_frqe = file_to_vector_array(train_signal, samplerate, n_mels=numcep, frames=frames, n_fft=1024,
                                              hop_length=512)

            if i == 0:
                frequency_spectrum_train = train_frqe
            else:
                frequency_spectrum_train = np.concatenate((frequency_spectrum_train, train_frqe), axis=0)  # 上下拼接

            i += 1

    data1 = frequency_spectrum_train
    data2 = frequency_spectrum_test

    return data1


def dnn_data_2test(datadir='../../../data/xingqing/data_dcase2020_task2/data/', type='fan', ID='id_00', frames=5,
                   numcep=128):
    frequency_spectrum_test_nor = []
    frequency_spectrum_test_anor = []
    datadir = datadir + type
    files2 = os.listdir(datadir + '/test')
    j = 0
    k = 0
    for everyone in files2:
        if ID in everyone:
            # print(everyone)
            test_signal, samplerate = librosa.load(datadir + '/test/' + everyone, sr=None, mono=False)
            test_frequency = file_to_vector_array(test_signal, samplerate, n_mels=numcep, frames=frames, n_fft=1024,
                                                  hop_length=512)

            if 'normal' in everyone:
                label = 1
                if j == 0:
                    frequency_spectrum_test_nor = test_frequency
                else:
                    frequency_spectrum_test_nor = np.concatenate((frequency_spectrum_test_nor, test_frequency),
                                                                 axis=0)  # 上下拼接
                j += 1
            else:
                label = 2
                if k == 0:
                    frequency_spectrum_test_anor = test_frequency
                else:
                    frequency_spectrum_test_anor = np.concatenate((frequency_spectrum_test_anor, test_frequency),
                                                                  axis=0)  # 上下拼接
                k += 1

    data2 = frequency_spectrum_test_nor
    data3 = frequency_spectrum_test_anor

    return data2, data3
#
#
# data_path_dev = '../../../data/xingqing/data_dcase2020_task2/data/dev/'
# for ty_pe in ['fan', 'slider', 'pump', 'valve', 'ToyCar', 'ToyConveyor']:
#     if ty_pe == 'ToyCar':
#         for ID in ['id_01', 'id_02', 'id_03', 'id_04']:
#             x_train = dnn_data_train(datadir=data_path_dev, type=ty_pe, ID=ID)
#             x_val, x_val_a = dnn_data_2test(datadir=data_path_dev, type=ty_pe, ID=ID)
#     elif ty_pe == 'ToyConveyor':
#         for ID in ['id_01', 'id_02', 'id_03']:
#             x_train = dnn_data_train(datadir=data_path_dev, type=ty_pe, ID=ID)
#             x_val, x_val_a = dnn_data_2test(datadir=data_path_dev, type=ty_pe, ID=ID)
#     else:
#         for ID in ['id_00', 'id_02', 'id_04', 'id_06']:
#             x_train = dnn_data_train(datadir=data_path_dev, type=ty_pe, ID=ID)
#             x_val, x_val_a = dnn_data_2test(datadir=data_path_dev, type=ty_pe, ID=ID)
