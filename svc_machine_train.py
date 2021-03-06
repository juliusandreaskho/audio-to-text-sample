# -*- coding: utf-8 -*-
"""SVC_machine_train.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/16hO-vvrwHitgIWQWeFmQhUd05CfQiQSQ

# Import Library (audio2numpy)
"""

!pip install audio2numpy

import os
import joblib
import numpy as np
from audio2numpy import open_audio
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score

"""#Import Library (w/struct and wave)"""

!pip install pygame

import os
import wave
import joblib
import struct
import numpy as np
import pygame.sndarray
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score

"""# Import Library (TQDM and soundfile)"""

!pip install soundfile

import os
import numpy as np
from tqdm import tqdm
import pandas as pd
import soundfile as sf
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score

"""# Array Declaration"""

words = [
            'baju', 'masker', 'sabun', 'meja', 'kursi', 'nasi', 'telur', 'ibu', 
            'bapak', 'cangkir', 'piring', 'sendok', 'garpu', 'mobil', 'kipas',
            'kompor', 'kakak', 'celana', 'sakit', 'demam', 'terik', 'matahari',
            'bulan', 'bintang', 'awan', 'kamu', 'pergi', 'datang', 'kembali',
            'dia', 'harus', 'anda', 'kami', 'saya', 'puasa', 'berlari',
            'berjalan', 'tidur', 'duduk', 'makan', 'minum', 'mandi', 'bangun',
            'pagi', 'malam', 'siang', 'sedang', 'dengan', 'membeli' ,'melihat'
        ]

"""# Data Read Function

Hope this could give better accuracy
"""

def read_training_data(training_directory):
    target_data = []
    sound_data = []
        
    for each_word in words:
      for each in [0,1,3,4,5,6,7,8]:
          sound_path = os.path.join(training_directory, each_word, each_word + '_' 
                                    + str(each) + '.wav')
          signal, sampling_rate = open_audio(sound_path)
          
          sound_data.append(signal)
          target_data.append(each_word)

    return (np.array(sound_data), np.array(target_data))

"""Read Data with cross-val score 2 - 2.5% and dataset accuracy 2.25% (9/400 ditebak benar [2 sakit, 7 bulan])"""

def read_training_data(training_directory):
    target_data = []
    sound_data = []
        
    for each_word in words:
      for each in [0,1,3,4,5,6,7,8]:
          sound_path = os.path.join(training_directory, each_word, each_word + '_' 
                                    + str(each) + '.wav')
          sound_read = wave.open(sound_path,'rb')
          length = sound_read.getnframes()
          for i in range (length):
            waveData = sound_read.readframes(1)
            data = struct.unpack("<1h", waveData)
          
          sound_data.append(data)
          target_data.append(each_word)

    return (pygame.sndarray.array(sound_data), np.array(target_data))

"""# Read Data 2"""

def read_training_data(training_directory):
    target_data = []
    sound_data = []
    # sound_col = []
    
    for each_word in words:
      for each in [0,1,3,4,5,6,7,8]:
        target_data.append(each_word)

    for each_word in words:
      for each in [0,1,3,4,5,6,7,8]:
          sound_path = os.path.join(training_directory, each_word, each_word + '_' 
                                    + str(each) + '.wav')
          sound_read = wave.open(sound_path,'rb')
          length = sound_read.getnframes()
          # for i in range (length):
          waveData = sound_read.readframes(1)
          data = struct.unpack("<1h", waveData)
          # ys = np.fromstring(waveData, dtype=np.int16)
          # data.shape()
          sound_data.append(data)

          # sound_data = np.reshape(sound_data, (1, 109440), order='C')
          # sound_data.shape()
          # target_data.shape()
    return (pygame.sndarray.array(sound_data), np.array(target_data))

def read_training_data(current_dir):
    audios = []
    target_data = []

    for each_word in words:
      path = current_dir + "/train_sound_mono/" + each_word + "/" 
      total = len(os.listdir(path))
      pbar = tqdm(total = total)
      for file in os.listdir(path):
        data, sr = sf.read(path + file)
        audios.append(data)
        target_data.append(each_word)
        pbar.update(1)
      pbar.close()
      
    return (np.array(audios), np.array(target_data))

"""# Cross Val Function

This uses the concept of cross validation to measure the accuracy
of a model, the num_of_fold determines the type of validation
num_of_fold is 5, we are performing a 5-fold cross validation
it will divide the dataset into 5 and use 1/5 of it for testing
and the remaining 4/5 for the training.
"""

def cross_validation(model, num_of_fold, train_data, train_label):
    accuracy_result = cross_val_score(model, train_data, train_label,
                                      cv=num_of_fold)
    print("Cross Validation Result for ", str(num_of_fold), " -fold")

    print(accuracy_result * 100)

"""# Path Declaration"""

current_dir = os.path.dirname(os.path.realpath("drive/My Drive/Colab Notebooks"))

training_dataset_dir = os.path.join(current_dir, 'train_sound_mono')

sound_data, target_data = read_training_data(training_dataset_dir)

print(sound_data.shape)
print(target_data.shape)

"""# Path Declare 2"""

current_dir = os.path.dirname(os.path.realpath("drive/My Drive/Colab Notebooks"))

sound_data, target_data = read_training_data(current_dir)

print(sound_data.shape)
print(target_data.shape)

"""# SVC Declaration"""

svc_model = LinearSVC(penalty='l2', loss='squared_hinge', dual=True, tol=1e-4, 
                      C=1.0, multi_class='ovr', fit_intercept=True, 
                      intercept_scaling=1, class_weight='balanced', verbose=0, 
                      random_state=None, max_iter=1000)

"""the kernel can be 'linear', 'poly', 'sigmoid' or 'rbf'
the probability was set to True so as to show
how sure the model is of it's prediction

svc_model = SVC(kernel='sigmoid', gamma='scale', shrinking=True, coef0=0.0, 
                probability=True, tol=1e-3, class_weight='balanced', max_iter=-1, 
                decision_function_shape = 'ovr', random_state=None)

# Cross Validation
"""

cross_validation(svc_model, 5, sound_data, target_data)

"""# SVM Data Fit

let's train the model with all the input data
"""

svc_model.fit(sound_data, target_data)

"""# Joblib Save Model

we will use the joblib module to persist the model
into files. This means that the next time we need to
predict, we don't need to train the model again
"""

save_directory = os.path.join(current_dir, 'models', 'svc')
if not os.path.exists(save_directory):
    os.makedirs(save_directory)
joblib.dump(svc_model, save_directory+'/svc.pkl')