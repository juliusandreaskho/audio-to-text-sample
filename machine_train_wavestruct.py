# import pyaudio
# import pygame.mixer
import os
import wave
import joblib
import struct
import numpy as np
import pygame.sndarray
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
# from scipy.io.wavfile import read
# from matplotlib import pyplot as plt

words = [
            'baju', 'masker', 'sabun', 'meja', 'kursi', 'nasi', 'telur', 'ibu', 
            'bapak', 'cangkir', 'piring', 'sendok', 'garpu', 'mobil', 'kipas',
            'kompor', 'kakak', 'celana', 'sakit', 'demam', 'terik', 'matahari',
            'bulan', 'bintang', 'awan', 'kamu', 'pergi', 'datang', 'kembali',
            'dia', 'harus', 'anda', 'kami', 'saya', 'puasa', 'berlari',
            'berjalan', 'tidur', 'duduk', 'makan', 'minum', 'mandi', 'bangun',
            'pagi', 'malam', 'siang', 'sedang', 'dengan', 'membeli' ,'melihat'
        ]

def read_training_data(training_directory):
    sound_data = []
    target_data = []
    # struct_data = []
    # sound_col = []
    # pygame.mixer.init(44100,-16,2,4096)
    
    for each_word in words:
        for each in [0,1,3,4,5,6,7,8]:
            sound_path = os.path.join(training_directory, each_word, each_word + '_' 
                                      + str(each) + '.wav')
            sound_read = wave.open(sound_path,'r')
            
            length = sound_read.getnframes()
            for i in range (length):
                waveData = sound_read.readframes(1) #how to get more than 1?
                data = struct.unpack("<1h", waveData)
                # struct_data[i+1] = data
                # data[i] = list (data)
                # data[i] = int(data)
                # sound_col.append(data)
            
            # flat_sound = np.reshape(sound_col, (-1), order='C')
            sound_data.append(data)
            target_data.append(each_word)
            
    # a = read(sound_data[180])
    # a = np.array(a[1],dtype=float)
    # plt.plot(a)
    # plt.show()
    
    return (pygame.sndarray.array(sound_data), np.array(target_data))
    
def cross_validation(model, num_of_fold, train_data, train_label):
    # this uses the concept of cross validation to measure the accuracy
    # of a model, the num_of_fold determines the type of validation
    # num_of_fold is 5, we are performing a 5-fold cross validation
    # it will divide the dataset into 5 and use 1/5 of it for testing
    # and the remaining 4/5 for the training
    accuracy_result = cross_val_score(model, train_data, train_label,
                                      cv=num_of_fold)
    print("Cross Validation Result for ", str(num_of_fold), " -fold")

    print(accuracy_result * 100)

current_dir = os.path.dirname(os.path.realpath(__file__))

training_dataset_dir = os.path.join(current_dir, 'train_sound_mono')

sound_data, target_data = read_training_data(training_dataset_dir)

svc_model = LinearSVC(penalty='l2', loss='squared_hinge', dual=True, tol=1e-4, 
                      C=1.0, multi_class='ovr', fit_intercept=True, 
                      intercept_scaling=1, class_weight='balanced', verbose=0, 
                      random_state=None, max_iter=1000)

# the kernel can be 'linear', 'poly', 'sigmoid' or 'rbf'
# the probability was set to True so as to show
# how sure the model is of it's prediction
"""
svc_model = SVC(kernel='sigmoid', gamma='scale', shrinking=True, coef0=0.0, 
                probability=True, tol=1e-3, class_weight='balanced', max_iter=-1, 
                decision_function_shape = 'ovr', random_state=None)
"""

cross_validation(svc_model, 5, sound_data, target_data)

# let's train the model with all the input data
svc_model.fit(sound_data, target_data)

# we will use the joblib module to persist the model
# into files. This means that the next time we need to
# predict, we don't need to train the model again
save_directory = os.path.join(current_dir, 'models', 'svc')
if not os.path.exists(save_directory):
    os.makedirs(save_directory)
joblib.dump(svc_model, save_directory+'\svc.pkl')