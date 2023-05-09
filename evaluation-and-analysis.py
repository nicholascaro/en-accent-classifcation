#!/usr/bin/env python
# coding: utf-8

# # Classification of Native and Non-Native English Accents

# In[1]:


import os
import IPython.display as ipd
import librosa
import matplotlib.pyplot as plt
import torch
import torchaudio
import torchaudio.transforms as TaudioT
import numpy as np
import random
import pandas as pd


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

# In[8]:


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

def mfcc(signal, n_mels=20):
    x, sr = signal
    melkwargs = {
        "n_fft": 512, "n_mels": n_mels, "hop_length": None, "mel_scale": "htk"
    }
    mfcc_transformer = TaudioT.MFCC(
        sample_rate = sr,
        n_mfcc = n_mels,
        melkwargs=melkwargs
    )
    mfcc_frames = mfcc_transformer(x)
    spec = TaudioT.AmplitudeToDB(top_db=80)(mfcc_frames)
    return spec


# Let's convert one audio sample to see what we are dealing with

# In[9]:


signal = torchaudio.load(wave_file)
signal = mono_to_stereo(signal)
signal = resample(signal)
signal = limit_length(signal)
mfcc_frame = mfcc(signal)

print(mfcc_frame.shape)

mfcc_frame


# In[10]:


def convert(wave, n_mels=20):
    signal = torchaudio.load(wave)
    signal = mono_to_stereo(signal)
    signal = resample(signal)
    signal = limit_length(signal)
    spec = mfcc(signal, n_mels=n_mels)
    return spec


# Let's convert the rest of the audio samples.  Since "welsh" has the lowest number of samples at 742, to keep balance in our dataset, we will convert 742 samples from each accent, and take the full 5936 from "american" for a perfectly balanced dataset for a binary classification predicting "american" vs "non-american" accents (8 x 742 = 5936)

# In[11]:


X_full = []
y_full = []

for accent in accents:
    count = 742 if accent != 'american' else 5936
    target_label = 0 if accent != 'american' else 1
    for file in os.listdir(data_path + accent + '/'):
        if file[0] != '.':
            wave_file_path = data_path + accent + '/' + file
            X_full.append(convert(wave_file_path))
            y_full.append(target_label)
            count -= 1
        if count == 0: break


# In[12]:


X_full = np.stack(X_full)
y_full = np.array(y_full)


# In[13]:


X_full.shape


# In[14]:


y_full.shape


# Save 4D numpy array and target vector to disk

# In[15]:


np.save('american-v-all-4D.npy', X_full)


# In[16]:


np.save('american-v-all-target.npy', y_full)


# Scale 4D data to [0, 1]

# In[17]:


X_min = X_full.min(axis=(0, 1), keepdims=True)
X_max = X_full.max(axis=(0, 1), keepdims=True)

X_full_norm = (X_full - X_min)/(X_max - X_min)


# In[18]:


print('x_full_norm:\n ', X_full_norm[0][0])


# Let's convert another set with a higher n_mel value to have a more detailed representation of each wave file and a higher dimensional data to work with for more powerful machine learning algorithms.  In addition we will attempt multiclass classification models predicting each accents available in our dataset.

# In[19]:


X_hd_full = []
y_hd_full = []

for accent in accents:
    count = 742
    for file in os.listdir(data_path + accent + '/'):
        if file[0] != '.':
            wave_file_path = data_path + accent + '/' + file
            X_hd_full.append(convert(wave_file_path, n_mels=64))
            y_hd_full.append(accents.index(accent))
            count -= 1
        if count == 0: break


# In[20]:


X_hd_full = np.stack(X_hd_full)
y_hd_full = np.stack(y_hd_full)


# In[21]:


print('X_hd_full: ', X_hd_full.shape)
print('y_hd_full: ', y_hd_full.shape)


# Let's make sure our target values reflect each class within our y vector

# In[22]:


index = 0
for i in range(9):
    index += i + 741 if i == 0 else 742
    print(f'For {accents[i]} the target value is {y_hd_full[index]}')


# Save 4D high dimension multiclass dataset onto disk

# In[ ]:


np.save('multiclass-4D-hd.npy', X_hd_full)


# In[ ]:


np.save('multiclass-4D-hd-target.npy', y_hd_full)


# Scale 4D high dimensional multiclass dataset to [0, 1]

# In[23]:


X_hd_min = X_hd_full.min(axis=(0, 1), keepdims=True)
X_hd_max = X_hd_full.max(axis=(0, 1), keepdims=True)

X_hd_full_norm = (X_hd_full - X_hd_min)/(X_hd_max - X_hd_min)


# In[24]:


print(X_hd_full_norm)


# Restructuring our dataset from 4d to 2d

# In[27]:


data = []
for audio in X_full:
    data.append(audio[0].flatten())

df = pd.DataFrame(data)
data = np.stack(data)


# In[28]:


data.shape


# Save 2D array as csv

# In[ ]:


np.save('american-v-all-2D.npy', data)


# ### Train Test Split

# In[29]:


from sklearn.model_selection import train_test_split


# Splitting our 2D data for non neural net model fitting

# In[30]:


X2d_train, X2d_test, y2d_train, y2d_test = train_test_split(data, y_full, test_size=0.15, random_state=42)


# Splitting our 4D data for neural net model fitting

# In[31]:


X_train, X_test, y_train, y_test = train_test_split(X_full_norm, y_full, test_size=0.15, random_state=42)


# In[32]:


print('X_train', X_train.shape)
print('y_train', y_train.shape)
print('X_test', X_test.shape)
print('y_test', y_test.shape)


# Splitting our 4D high dimensional data for more powerful neural networks

# In[33]:


X_hd_train, X_hd_test, y_hd_train, y_hd_test = train_test_split(X_hd_full_norm, y_hd_full, test_size=0.15, random_state=42)


# In[34]:


print('X_hd_train', X_hd_train.shape)
print('y_hd_train', y_hd_train.shape)
print('X_hd_test', X_hd_test.shape)
print('y_hd_test', y_hd_test.shape)


# ### Dimensionality Reduction of 2D data

# #### PCA

# In[ ]:


from sklearn.decomposition import PCA


# Peform principal component analysis while retaining 95% variance over our data

# In[ ]:


pca = PCA(n_components=0.95)


# #### Random Forest as Feature Selection

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel


# In[ ]:


rf_feature_select = SelectFromModel(RandomForestClassifier(n_estimators=100, max_leaf_nodes=9, n_jobs=-1))


# ### Combining Both Techniques

# In[ ]:


from sklearn.pipeline import Pipeline


# In[ ]:


full_pipeline = Pipeline(
    [('pca', pca), 
     ('rf', rf_feature_select)]
)

X2d_reduced = full_pipeline.fit_transform(X2d_train, y2d_train)


# In[ ]:


X2d_reduced.shape


# Performing random forest as feature selection after having completed principal component analysis brought down the dimension of our data from 5160 to 379.  We will keep this setting as we explore the different models available to us.

# ## Exploring Different Models and Applying Performance Evaluation Metrics

# ### Logistic Regression

# In[47]:


from sklearn.linear_model import LogisticRegression
from sklearn import metrics


# In[ ]:


lr_model = Pipeline([
    ('pca', pca),
    ('rf', rf_feature_select),
    ('lr', LogisticRegression())
])

lr_model.fit(X2d_train, y2d_train)


# In[ ]:


lr_predict = lr_model.predict(X2d_test)


# Classifcation Report, Confusion Matrix, ROC AUC Curve, and Precision - Recall Curve for Linear Regression Model

# In[ ]:


# classifcation report
lr_cr = metrics.classification_report(y2d_test, lr_predict)

# confusion matrix
lr_cm = metrics.confusion_matrix(y2d_test, lr_predict)
lr_cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=lr_cm, display_labels= lr_model.classes_)

# roc auc curve 
lr_fpr, lr_tpr, lr_thresholds = metrics.roc_curve(y2d_test, lr_predict)
lr_roc_auc = metrics.auc(lr_fpr, lr_tpr)
lr_roc_auc_display = metrics.RocCurveDisplay(fpr=lr_fpr, tpr=lr_tpr, roc_auc=lr_roc_auc)

# precision recall curve
lr_precision, lr_recall, _ = metrics.precision_recall_curve(y2d_test, lr_predict)
lr_prec_reca_display = metrics.PrecisionRecallDisplay(precision=lr_precision, recall=lr_recall)


# In[ ]:


# classifcation report
print(lr_cr)

# confusion matrix display
lr_cm_display.plot()

# roc auc curve display
lr_roc_auc_display.plot()

# precision recall curve display
lr_prec_reca_display.plot()


# ### K-nearest Neighbors (KNN) Classifer

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier


# In[ ]:


knn_model = Pipeline([
    ('pca', pca),
    ('rf', rf_feature_select),
    ('knn', KNeighborsClassifier(n_neighbors=7))
])

knn_model.fit(X2d_train, y2d_train)


# In[ ]:


knn_predict = knn_model.predict(X2d_test)


# Classifcation Report, Confusion Matrix, ROC AUC Curve, and Precision - Recall Curve for KNN Model

# In[ ]:


# classifcation report
knn_cr = metrics.classification_report(y2d_test, knn_predict)

# confusion matrix
knn_cm = metrics.confusion_matrix(y2d_test, knn_predict)
knn_cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=knn_cm, display_labels= knn_model.classes_)

# roc auc curve 
knn_fpr, knn_tpr, knn_thresholds = metrics.roc_curve(y2d_test, knn_predict)
knn_roc_auc = metrics.auc(knn_fpr, knn_tpr)
knn_roc_auc_display = metrics.RocCurveDisplay(fpr=knn_fpr, tpr=knn_tpr, roc_auc=knn_roc_auc)

# precision recall curve
knn_precision, knn_recall, _ = metrics.precision_recall_curve(y2d_test, knn_predict)
knn_prec_reca_display = metrics.PrecisionRecallDisplay(precision=knn_precision, recall=knn_recall)


# In[ ]:


# classifcation report
print(knn_cr)

# confusion matrix display
knn_cm_display.plot()

# roc auc curve display
knn_roc_auc_display.plot()

# precision recall curve display
knn_prec_reca_display.plot()


# ### Support Vector Machines (SVM) Classifier

# In[ ]:


from sklearn.svm import SVC


# In[ ]:


svm_model = Pipeline([
    ('pca', pca),
    ('rf', rf_feature_select),
    ('svm', SVC())
])

svm_model.fit(X2d_train, y2d_train)


# In[ ]:


svm_predict = svm_model.predict(X2d_test)


# Classifcation Report, Confusion Matrix, ROC AUC Curve, and Precision - Recall Curve for SVM Model

# In[ ]:


# classifcation report
svm_cr = metrics.classification_report(y2d_test, svm_predict)

# confusion matrix
svm_cm = metrics.confusion_matrix(y2d_test, svm_predict)
svm_cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=svm_cm, display_labels= svm_model.classes_)

# roc auc curve 
svm_fpr, svm_tpr, svm_thresholds = metrics.roc_curve(y2d_test, svm_predict)
svm_roc_auc = metrics.auc(svm_fpr, svm_tpr)
svm_roc_auc_display = metrics.RocCurveDisplay(fpr=svm_fpr, tpr=svm_tpr, roc_auc=svm_roc_auc)

# precision recall curve
svm_precision, svm_recall, _ = metrics.precision_recall_curve(y2d_test, svm_predict)
svm_prec_reca_display = metrics.PrecisionRecallDisplay(precision=svm_precision, recall=svm_recall)


# In[ ]:


# classifcation report
print(svm_cr)

# confusion matrix display
svm_cm_display.plot()

# roc auc curve display
svm_roc_auc_display.plot()

# precision recall curve display
svm_prec_reca_display.plot()


# ### Decision Trees Classifier

# In[ ]:


from sklearn.tree import DecisionTreeClassifier


# In[ ]:


tree_model = Pipeline([
    ('pca', pca),
    ('rf', rf_feature_select),
    ('tree', DecisionTreeClassifier(max_leaf_nodes=50))
])

tree_model.fit(X2d_train, y2d_train)


# In[ ]:


tree_predict = tree_model.predict(X2d_test)


# Classifcation Report, Confusion Matrix, ROC AUC Curve, and Precision - Recall Curve for Decision Tree Model

# In[ ]:


# classifcation report
tree_cr = metrics.classification_report(y2d_test, tree_predict)

# confusion matrix
tree_cm = metrics.confusion_matrix(y2d_test, tree_predict)
tree_cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=tree_cm, display_labels= tree_model.classes_)

# roc auc curve 
tree_fpr, tree_tpr, tree_thresholds = metrics.roc_curve(y2d_test, tree_predict)
tree_roc_auc = metrics.auc(tree_fpr, tree_tpr)
tree_roc_auc_display = metrics.RocCurveDisplay(fpr=tree_fpr, tpr=tree_tpr, roc_auc=tree_roc_auc)

# precision recall curve
tree_precision, tree_recall, _ = metrics.precision_recall_curve(y2d_test, tree_predict)
tree_prec_reca_display = metrics.PrecisionRecallDisplay(precision=tree_precision, recall=tree_recall)


# In[ ]:


# classifcation report
print(tree_cr)

# confusion matrix display
tree_cm_display.plot()

# roc auc curve display
tree_roc_auc_display.plot()

# precision recall curve display
tree_prec_reca_display.plot()


# ### Random Forests Classifier

# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:


forest_model = Pipeline([
    ('pca', pca),
    ('rf', rf_feature_select),
    ('tree', RandomForestClassifier(n_estimators=500, max_leaf_nodes=50, n_jobs=-1))
])

forest_model.fit(X2d_train, y2d_train)


# In[ ]:


forest_predict = forest_model.predict(X2d_test)


# Classifcation Report, Confusion Matrix, ROC AUC Curve, and Precision - Recall Curve for Random Forest Model

# In[ ]:


# classifcation report
forest_cr = metrics.classification_report(y2d_test, forest_predict)

# confusion matrix
forest_cm = metrics.confusion_matrix(y2d_test, forest_predict)
forest_cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=forest_cm, display_labels= forest_model.classes_)

# roc auc curve 
forest_fpr, forest_tpr, forest_thresholds = metrics.roc_curve(y2d_test, forest_predict)
forest_roc_auc = metrics.auc(forest_fpr, forest_tpr)
forest_roc_auc_display = metrics.RocCurveDisplay(fpr=forest_fpr, tpr=forest_tpr, roc_auc=forest_roc_auc)

# precision recall curve
forest_precision, forest_recall, _ = metrics.precision_recall_curve(y2d_test, forest_predict)
forest_prec_reca_display = metrics.PrecisionRecallDisplay(precision=forest_precision, recall=forest_recall)


# In[ ]:


# classifcation report
print(forest_cr)

# confusion matrix display
forest_cm_display.plot()

# roc auc curve display
forest_roc_auc_display.plot()

# precision recall curve display
forest_prec_reca_display.plot()


# ### AdaBoost Classifier

# In[ ]:


from sklearn.ensemble import AdaBoostClassifier


# In[ ]:


ada_clf = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=1), n_estimators=500, algorithm="SAMME.R", learning_rate=0.1
)

ada_model = Pipeline([
    ('pca', pca),
    ('rf', rf_feature_select),
    ('ada', ada_clf)
])

ada_model.fit(X2d_train, y2d_train)


# In[ ]:


ada_predict = ada_model.predict(X2d_test)


# Classifcation Report, Confusion Matrix, ROC AUC Curve, and Precision - Recall Curve for Ada Boost  Model

# In[ ]:


# classifcation report
ada_cr = metrics.classification_report(y2d_test, ada_predict)

# confusion matrix
ada_cm = metrics.confusion_matrix(y2d_test, ada_predict)
ada_cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=ada_cm, display_labels= ada_model.classes_)

# roc auc curve 
ada_fpr, ada_tpr, ada_thresholds = metrics.roc_curve(y2d_test, ada_predict)
ada_roc_auc = metrics.auc(ada_fpr, ada_tpr)
ada_roc_auc_display = metrics.RocCurveDisplay(fpr=ada_fpr, tpr=ada_tpr, roc_auc=ada_roc_auc)

# precision recall curve
ada_precision, ada_recall, _ = metrics.precision_recall_curve(y2d_test, ada_predict)
ada_prec_reca_display = metrics.PrecisionRecallDisplay(precision=ada_precision, recall=ada_recall)


# In[ ]:


# classifcation report
print(ada_cr)

# confusion matrix display
ada_cm_display.plot()

# roc auc curve display
ada_roc_auc_display.plot()

# precision recall curve display
ada_prec_reca_display.plot()


# ### Gradient Boosting Classifer

# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier


# In[ ]:


gdbst = GradientBoostingClassifier(
    max_depth=2, n_estimators=500, learning_rate=0.1
)

gdbst_model = Pipeline([
    ('pca', pca),
    ('rf', rf_feature_select),
    ('gd', gdbst)
])

gdbst_model.fit(X2d_train, y2d_train)


# In[ ]:


gdbst_predict = gdbst_model.predict(X2d_test)


# Classifcation Report, Confusion Matrix, ROC AUC Curve, and Precision - Recall Curve for Gradient Boost  Model

# In[ ]:


# classifcation report
gdbst_cr = metrics.classification_report(y2d_test, gdbst_predict)

# confusion matrix
gdbst_cm = metrics.confusion_matrix(y2d_test, gdbst_predict)
gdbst_cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=gdbst_cm, display_labels= gdbst_model.classes_)

# roc auc curve 
gdbst_fpr, gdbst_tpr, gdbst_thresholds = metrics.roc_curve(y2d_test, gdbst_predict)
gdbst_roc_auc = metrics.auc(gdbst_fpr, gdbst_tpr)
gdbst_roc_auc_display = metrics.RocCurveDisplay(fpr=gdbst_fpr, tpr=gdbst_tpr, roc_auc=gdbst_roc_auc)

# precision recall curve
gdbst_precision, gdbst_recall, _ = metrics.precision_recall_curve(y2d_test, gdbst_predict)
gdbst_prec_reca_display = metrics.PrecisionRecallDisplay(precision=gdbst_precision, recall=gdbst_recall)


# In[ ]:


# classifcation report
print(gdbst_cr)

# confusion matrix display
gdbst_cm_display.plot()

# roc auc curve display
gdbst_roc_auc_display.plot()

# precision recall curve display
gdbst_prec_reca_display.plot()


# ## Artificial Neural Networks

# ### Perceptron

# In[ ]:


from sklearn.linear_model import Perceptron


# In[ ]:


per_model = Pipeline([
    ('pca', pca),
    ('rf', rf_feature_select),
    ('per', Perceptron(n_jobs=-1))
])

per_model.fit(X2d_train, y2d_train)


# In[ ]:


per_predict = per_model.predict(X2d_test)


# Classifcation Report, Confusion Matrix, ROC AUC Curve, and Precision - Recall Curve for Perceptron Model

# In[ ]:


# classifcation report
per_cr = metrics.classification_report(y2d_test, per_predict)

# confusion matrix
per_cm = metrics.confusion_matrix(y2d_test, per_predict)
per_cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=per_cm, display_labels= per_model.classes_)

# roc auc curve 
per_fpr, per_tpr, per_thresholds = metrics.roc_curve(y2d_test, per_predict)
per_roc_auc = metrics.auc(per_fpr, per_tpr)
per_roc_auc_display = metrics.RocCurveDisplay(fpr=per_fpr, tpr=per_tpr, roc_auc=per_roc_auc)

# precision recall curve
per_precision, per_recall, _ = metrics.precision_recall_curve(y2d_test, per_predict)
per_prec_reca_display = metrics.PrecisionRecallDisplay(precision=per_precision, recall=per_recall)


# In[ ]:


# classifcation report
print(per_cr)

# confusion matrix display
per_cm_display.plot()

# roc auc curve display
per_roc_auc_display.plot()

# precision recall curve display
per_prec_reca_display.plot()


# ### Multilayer Perceptron (MLP) Binary Classification

# In[35]:


import tensorflow as tf
from tensorflow import keras


# In[86]:


mlp_binary_model = keras.models.Sequential()
mlp_binary_model.add(keras.layers.Flatten(input_shape=[2, 20, 258]))
mlp_binary_model.add(keras.layers.Dense(300, activation="relu"))
mlp_binary_model.add(keras.layers.Dense(300, activation="relu"))
mlp_binary_model.add(keras.layers.Dense(300, activation="relu"))
mlp_binary_model.add(keras.layers.Dense(1, activation="sigmoid"))


# In[87]:


mlp_binary_model.summary()


# In[88]:


mlp_binary_model.compile(
    loss='binary_crossentropy',
    optimizer=keras.optimizers.SGD(learning_rate=0.01, momentum=0.9), 
    metrics=["accuracy"]
)


# In[91]:


early_stopping_cb = keras.callbacks.EarlyStopping(patience=10)

checkpoint_cb = keras.callbacks.ModelCheckpoint("mlp_binary.h5", save_best_only = True)

mlp_binary_history = mlp_binary_model.fit(
    X_train, y_train, epochs=100,
    validation_split=0.1,
    callbacks=[checkpoint_cb, early_stopping_cb]
)

mlp_binary_model = keras.models.load_model("mlp_binary.h5")


# In[92]:


mlp_binary_model.evaluate(X_test, y_test)


# In[98]:


mlp_predict = (mlp_binary_model.predict(X_test) > 0.5).astype("int32")
# mlp_predict = (mlp_model.predict(X_test) > 0.5).astype("int32")


# Classifcation Report, Confusion Matrix, ROC AUC Curve, and Precision - Recall Curve for Multilayer Perceptron Binary Classification

# In[100]:


# classifcation report
mlp_cr = metrics.classification_report(y_test, mlp_predict)

# confusion matrix
mlp_cm = metrics.confusion_matrix(y_test, mlp_predict)
mlp_cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=mlp_cm)

# roc auc curve 
mlp_fpr, mlp_tpr, mlp_thresholds = metrics.roc_curve(y_test, mlp_predict)
mlp_roc_auc = metrics.auc(mlp_fpr, mlp_tpr)
mlp_roc_auc_display = metrics.RocCurveDisplay(fpr=mlp_fpr, tpr=mlp_tpr, roc_auc=mlp_roc_auc)

# precision recall curve
mlp_precision, mlp_recall, _ = metrics.precision_recall_curve(y_test, mlp_predict)
mlp_prec_reca_display = metrics.PrecisionRecallDisplay(precision=mlp_precision, recall=mlp_recall)


# In[101]:


# classifcation report
print(mlp_cr)

# confusion matrix display
mlp_cm_display.plot()

# roc auc curve display
mlp_roc_auc_display.plot()

# precision recall curve display
mlp_prec_reca_display.plot()


# Multilayer Preceptron (MLP) Multiclass Classification

# In[103]:


mlp_model = keras.models.Sequential()
mlp_model.add(keras.layers.Flatten(input_shape=[2, 64, 258]))
mlp_model.add(keras.layers.Dense(300, activation="relu"))
mlp_model.add(keras.layers.Dense(300, activation="relu"))
mlp_model.add(keras.layers.Dense(300, activation="relu"))
mlp_model.add(keras.layers.Dense(9, activation="softmax"))


# In[104]:


mlp_model.summary()


# In[105]:


mlp_model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=keras.optimizers.SGD(learning_rate=0.01, momentum=0.9), 
    metrics=["accuracy"]
)


# In[106]:


early_stopping_cb = keras.callbacks.EarlyStopping(patience=10)

checkpoint_cb = keras.callbacks.ModelCheckpoint("mlp_multiclass.h5", save_best_only = True)

mlp_history = mlp_model.fit(
    X_hd_train, y_hd_train, epochs=100,
    validation_split=0.1,
    callbacks=[checkpoint_cb, early_stopping_cb]
)

mlp_model = keras.models.load_model("mlp_multiclass.h5")


# In[107]:


mlp_model.evaluate(X_hd_test, y_hd_test)


# In[108]:


mlp_prob = mlp_model.predict(X_hd_test)
mlp_predict = mlp_prob.argmax(axis=-1)


# In[109]:


# classifcation report
mlp_cr = metrics.classification_report(y_hd_test, mlp_predict)

# confusion matrix
mlp_cm = metrics.multilabel_confusion_matrix(y_hd_test, mlp_predict)
mlp_cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=mlp_cm)


# In[126]:


# classifcation report
print(mlp_cr)


# ### Convolutional Neural Networks (CNN)

# In[111]:


X_hd_train_reshaped = np.moveaxis(X_hd_train, 1, -1)
X_hd_test_reshaped = np.moveaxis(X_hd_test, 1, -1)


# In[112]:


print("X_reshaped: ", X_hd_train_reshaped.shape)
print("X_train_reshaped: ", X_hd_test_reshaped.shape)


# In[113]:


cnn_model = keras.models.Sequential([
    keras.layers.Conv2D(128, 7, strides=2, activation="relu", padding="same", input_shape=[64, 258, 2]),
    keras.layers.MaxPooling2D(2),
    keras.layers.Conv2D(256, 3, strides=2, activation="relu", padding="same"),
    keras.layers.MaxPooling2D(2),
    keras.layers.Flatten(),
    keras.layers.Dense(256, activation="relu"),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(9, activation="softmax"),
])


# In[114]:


cnn_model.summary()


# In[115]:


cnn_model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer="adam", 
    metrics=["accuracy"]
)


# In[ ]:


early_stopping_cb = keras.callbacks.EarlyStopping(patience=10)

checkpoint_cb = keras.callbacks.ModelCheckpoint("cnn_multiclass.h5", save_best_only = True)

cnn_history = cnn_model.fit(
    X_hd_train_reshaped, y_hd_train, epochs=100,
    validation_split=0.1,
    callbacks=[checkpoint_cb, early_stopping_cb]
)

cnn_model = keras.models.load_model("cnn_multiclass.h5")


# In[117]:


cnn_model.evaluate(X_hd_test_reshaped, y_hd_test)


# In[121]:


cnn_prob = cnn_model.predict(X_hd_test_reshaped)
cnn_predict = cnn_prob.argmax(axis=-1)


# In[124]:


# classifcation report
cnn_cr = metrics.classification_report(y_hd_test, cnn_predict)

# confusion matrix
cnn_cm = metrics.multilabel_confusion_matrix(y_hd_test, cnn_predict)
cnn_cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=cnn_cm)


# In[125]:


# classifcation report
print(cnn_cr)

