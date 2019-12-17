import time
import pickle
import jax
import numpy as np
import sys
sys.path.insert(0, "../TheanoXLA")
from scipy.io.wavfile import read
import glob
import theanoxla
import theanoxla.tensor as T
from theanoxla import layers, function

import matplotlib.pyplot as plt
from matplotlib import interactive
interactive(False)
#https://github.com/google/jax/blob/master/jax/lib/xla_bridge.py
from jax.lib import xla_client
from sklearn.metrics import roc_auc_score, accuracy_score




time = T.linspace(-5, 5, 100)

f1 = T.Placeholder((), 'float32')
f2 = T.Placeholder((), 'float32')

filter = 2 *(f1 * T.sinc(time*f1) - f2 * T.sinc(time*f2)) * T.signal.hanning(100)

get_filter = function(f1, f2, outputs=[filter])

plt.plot(get_filter(1.5, 1.3)[0])
plt.show()

