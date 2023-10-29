import matplotlib.pyplot as plt
import numpy as np
import keras
import tensorflow as tf
from keras.datasets import mnist
from itertools import islice
import random
import time

from mlp import Dense
from loss import softmax_crossentropy_with_logits, grad_softmax_crossentropy_with_logits
from back_and_forth import forward, predict, train
from get_data import get_data, iterate_minibatches
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
