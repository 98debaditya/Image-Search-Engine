import numpy as np
import tensorflow as tf
import os
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from keras.applications.vgg16 import decode_predictions
from tensorflow.keras.models import Model

def Resize(X):
    X = X.transpose(2,0,1)
    A = cv2.resize(X[0], (224,224))
    B = cv2.resize(X[1], (224,224))
    C = cv2.resize(X[2], (224,224))
    X = np.concatenate([[A],[B],[C]])
    return X.transpose(1,2,0)

def load_images(X):
    lst = os.listdir(X)
    image_list = []
    for filename in lst:
        img = Image.open(os.path.join(X, filename))
        img = np.array(img)
        img = Resize(img)
        image_list = image_list + [img]
    return np.array(image_list)

def Predict(X,model):
    X = X.reshape((1, X.shape[0], X.shape[1], X.shape[2]))
    X = tf.keras.applications.vgg16.preprocess_input(X)
    pred = model.predict(X)
    return pred

def Vectors(X,model):
    X = load_images(X)
    L = X.shape[0]
    vec = []
    for i in range(0,L):
        X_i = X[i]
        vec = vec + [Predict(X_i,model)]
    vec = np.array(vec)
    return vec

model = tf.keras.applications.VGG16(weights='imagenet')
model = Model(inputs=model.input, outputs=model.get_layer('fc1').output)
path = '/home/deb/quary/phone'
vec = Vectors(path,model)
np.save('/home/deb/quary/vec.npy', vec)
