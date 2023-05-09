#!/usr/bin/env python
# coding: utf-8

# # Classification of Native and Non-Native English Accents

# In[14]:


import os
import IPython.display as ipd
import librosa
# import librosa.display
import matplotlib.pyplot as plt
import torch
import torchaudio
import torchaudio.transforms as TaudioT
import numpy as np
import random


# ## Data Exploration

# The different accents we have in our dataset

# In[2]:


data_path = './accentdb_extended_combined/data/'
accents = [accent for accent in os.listdir(data_path) if accent[0] != '.']
print(accents)
print('count: ', len(accents))


# The number of audio samples we have for each accents

# In[3]:


for accent in accents:
    files = [file for file in os.listdir(data_path + accent + '/') if file[0] != '.']
    print((accent, len(files)))


# Let's take a sample wave file and listen to what the recording sounds like.

# In[4]:


wave_file = "./accentdb_extended_combined/data/indian/indian_s01_008.wav"
# Audio is part of IPython's disply module providing audio controls
ipd.Audio(filename=wave_file)


# Let's see a pressure vs time graph of the wave file

# In[5]:


# librosa is a python package for music and audio analysis
# load converts a wave file to audio time series numpy array
# return:
# x -> numpy ndarray; audio time series multi channel supported
# sr -> scalar value; sample rate
x, sr = librosa.load(wave_file)
plt.figure(figsize=(14, 5))
librosa.display.waveshow(x, sr=sr)
plt.show()


# Mel-frequency cepstral coefficients (MFCC) is known in the audio signal analisys field to be the best representation for human speech audio signals.  Let's convert this wave file into an MFCC frame, and see what it looks like.

# In[6]:


# librosa's feature extraction module mfcc converter
# returns a numpy array of mfcc sequence
mfccs = librosa.feature.mfcc(y=x, sr=sr, n_mfcc=30)
print(mfccs.shape)

plt.figure(figsize=(14, 7))
librosa.display.specshow(mfccs, sr=sr, x_axis='time')
plt.show()


# Exploring meta data

# In[7]:


for accent in accents:
    path = data_path + accent + '/'
    print(accent)
    directory = [file for file in os.listdir(path) if file[0] != '.']
    for file in directory:
        file_path = path + file
        print(torchaudio.info(file_path))
        x, sr = librosa.load(file_path)
        plt.figure(figsize=(14, 5))
        librosa.display.waveshow(x, sr=sr)
        plt.show()
        break


# We see difference in sample rate (22050 vs 48000) and number of channels (mono vs stereo) which translates into difference in dimensions of resulting MFCC representation which is an issue for our modeling process that we need to address.

# In[6]:


# helper functions
def mono_to_stereo(signal):
    x, sr = signal
    if x.shape[0] == 2:
        return signal
    stereo = torch.cat([x, x])
    return stereo, sr

def resample(signal, new_sr=22050):
    x, sr = signal
    if sr == new_sr:
        return signal
    channel_1 = TaudioT.Resample(sr, new_sr)(x[:1, :])
    channel_2 = TaudioT.Resample(sr, new_sr)(x[1:, :])
    new_x = torch.cat([channel_1, channel_2])
    return new_x, new_sr

def limit_length(sig, ms=3000):
    x, sr = sig
    rows, audio_len = x.shape
    max_len = sr // 1000 * ms

    if audio_len > max_len:
        x = x[:, :max_len]
    elif audio_len < max_len:
        diff = max_len - audio_len
        append_start_len = random.randint(0, diff)
        append_stop_len = diff - append_start_len
        append_start = torch.zeros((rows, append_start_len))
        append_stop = torch.zeros((rows, append_stop_len))

        x = torch.cat((append_start, x, append_stop), 1)
    return x, sr

def mfcc(signal):
    x, sr = signal
    melkwargs = {
        "n_fft": 512, "n_mels": 20, "hop_length": None, "mel_scale": "htk"
    }
    mfcc_transformer = TaudioT.MFCC(
        sample_rate = sr,
        n_mfcc = 20,
        melkwargs=melkwargs
    )
    mfcc_frames = mfcc_transformer(x)
    spec = TaudioT.AmplitudeToDB(top_db=80)(mfcc_frames)
    return spec


# Let's convert one audio sample to see what we are dealing with

# In[7]:


signal = torchaudio.load(wave_file)
signal = mono_to_stereo(signal)
signal = resample(signal)
signal = limit_length(signal)
mfcc_frame = mfcc(signal)

print(mfcc_frame.shape)

mfcc_frame


# Let's convert the rest of the audio samples.  Since "welsh" has the lowest number of samples at 742, to keep balance in our dataset, we will convert 742 samples from each accent.

# In[99]:


def convert(wave):
    signal = torchaudio.load(wave)
    signal = mono_to_stereo(signal)
    signal = resample(signal)
    signal = limit_length(signal)
    spec = mfcc(signal)
    return spec

X_full = []
y_full = []

for accent in accents:
    count = 742
    for file in os.listdir(data_path + accent + '/'):
        if file[0] != '.':
            wave_file_path = data_path + accent + '/' + file
            X_full.append(convert(wave_file_path))
            y_full.append(accents.index(accent))
            count -= 1
        if count == 0: break


# In[101]:


X_full = np.stack(X_full)
y_full = np.array(y_full)


# In[103]:


X_full.shape


# In[104]:


y_full.shape


# ## Modeling using SVM

# In[125]:


from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.decomposition import IncrementalPCA
from sklearn.metrics import accuracy_score
import pickle


# Restructuring our dataset from 4d to 2d

# In[85]:


data = []
for audio in X_full:
    data.append(audio[0].flatten())

data = np.stack(data)


# In[86]:


data.shape


# Splitting training and testing set with full feature set (pre feature selection / dimensionality reduction)

# In[105]:


Xf_train, Xf_test, yf_train, yf_test = train_test_split(data, y_full, test_size=0.15, random_state=42)


# In[106]:


print('Xf_train: ', Xf_train.shape)
print('Xf_test: ', Xf_test.shape)
print('yf_train: ', yf_train.shape)
print('yf_test: ', yf_test.shape)


# Model fit using full feature set

# In[107]:


svm_full = make_pipeline(StandardScaler(), SVC(gamma='auto'))
svm_full.fit(Xf_train, yf_train)


# In[126]:


filename = 'finalized_model_M1.sav'
pickle.dump(svm_full, open(filename, 'wb'))

# load the model from disk
loaded_model_full = pickle.load(open(filename, 'rb'))
result = loaded_model_full.score(Xf_test,yf_test)


# In[127]:


print(result)


# ### Dimensionality Reduction

# #### Principal Component Analysis (PCA)

# In[21]:


pca = PCA(n_components=0.95)
X_reduced = pca.fit_transform(data)


# In[22]:


X_reduced.shape


# #### Incremental PCA

# In[48]:


n_batches = 100 
inca_pca = IncrementalPCA(n_components=66)
for X_batch in np.array_split(X_reduced, n_batches):
    inca_pca.partial_fit(X_batch)
X_inca = inca_pca.transform(X_reduced)


# In[49]:


X_inca.shape


# Separating training and testing set with reduced feature set (post dimensionality reduction)

# In[119]:


Xr_train, Xr_test, Yr_train, Yr_test = train_test_split(X_inca, y_full, test_size=0.15, random_state=42)


# In[121]:


print('Xr_train: ', Xr_train.shape)
print('Xr_test: ', Xr_test.shape)
print('Yr_train: ', yr_train.shape)
print('Yr_test: ', yr_test.shape)


# Model fitting using reduced feature set

# In[122]:


svm_reduced = make_pipeline(StandardScaler(), SVC(gamma='auto'))
svm_reduced.fit(Xr_train, Yr_train)


# In[152]:


filename = 'finalized_model_M2.sav'
pickle.dump(svm_reduced, open(filename, 'wb'))

# load the model from disk
loaded_model_r = pickle.load(open(filename, 'rb'))
result = loaded_model_r.score(Xr_test,Yr_test)


# In[153]:


print(result)


# #### SVM with Dimensionality Reduction after Train-Test Split

# In[133]:


X_train, X_test, Y_train, Y_test = train_test_split(data, y_full, test_size=0.15, random_state=42)


# PCA

# In[135]:


pca = PCA(n_components=0.95)
X_train_pca = pca.fit_transform(X_train)


# In[136]:


X_train_pca.shape


# Incremental PCA

# In[140]:


n_batches = 100 
inca_pca = IncrementalPCA(n_components=56)
for X_batch in np.array_split(X_train_pca, n_batches):
    inca_pca.partial_fit(X_batch)
X_train_inca = inca_pca.transform(X_train_pca)


# In[141]:


X_train_inca.shape


# In[142]:


svm = make_pipeline(StandardScaler(), SVC(gamma='auto'))
svm.fit(X_train, Y_train)


# In[143]:


pca = PCA(n_components=0.95)
X_test_pca = pca.fit_transform(X_test)


# In[144]:


X_test_pca.shape


# In[150]:


n_batches = 5
inca_pca = IncrementalPCA(n_components=56)
for X_batch in np.array_split(X_test_pca, n_batches):
    inca_pca.partial_fit(X_batch)
X_test_inca = inca_pca.transform(X_test_pca)


# In[151]:


X_test_inca.shape


# In[155]:


filename = 'finalized_model_M3.sav'
pickle.dump(svm, open(filename, 'wb'))

# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(X_test,Y_test)


# In[156]:


print(result)

