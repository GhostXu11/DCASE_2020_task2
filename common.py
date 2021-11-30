import glob
import argparse
import sys
import os

import numpy as np
import librosa
import librosa.core
import librosa.feature
import math
import scipy.sparse as ss
import yaml
import logging

logging.basicConfig(level=logging.DEBUG, filename="baseline.log")
logger = logging.getLogger(' ')
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)