import sys
import librosa
import numpy as np
import os
import scipy.io.wavfile as wav
from python_speech_features import *


def one_dimentional_data(data_dir = './data/', type = 'fan', ID = 'id_00', winlen=0.064, winstep=0.032, numcep=128,ratio_train=1):
    frequency_spectrum_train = []
    frequency_spectrum_test_nor = []
    frequency_spectrum_test_anor = []
