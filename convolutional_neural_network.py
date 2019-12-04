import numpy as np
import os
import PIL
from matplotlib import pyplot as plt
import cv2
import random
import pickle
import glob
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import pickle
from keras.models import model_from_json
from keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

file_list = []
class_list = []

DATADIR = "dataset"

CATEGORIES = ["uncompressed"]

IMG_SIZE = 50

for category in CATEGORIES:
    path = os.path.join(DATADIR, category)
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)

training_data = []
image_list = []
for category in CATEGORIES:
    class_num = CATEGORIES.index(category)
    for filename in glob.glob('dataset/uncompressed/*.gif'):
        try:
            img = PIL.Image.open(filename).convert("L")
            imgarr = np.array(img)
            image_list.append(imgarr)
            training_data.append([imgarr, class_num])
        except Exception as e:
            pass

random.shuffle(training_data)

X = [] #features
y = [] #labels

for features, label in training_data:
    X.append(features)
    y.append(label)

X = np.array(X).reshape(-1, 30, 30, 1)
y = np.asarray(y)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# normalizing data (a pixel goes from 0 to 255)
X = X/255.0

# Building the model
model = Sequential()

# 3 convolutional layers
model.add(Conv2D(32, (3, 3), input_shape=X.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation("relu"))

model.add(Dense(128))
model.add(Activation("relu"))

# The output layer with 13 neurons, for 13 classes
model.add(Dense(13))
model.add(Activation("softmax"))

# Compiling the model using some basic parameters
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Training the model, with 40 iterations
# validation_split corresponds to the percentage of images used for the validation phase compared to all the images
# print(X)
# print(len(X[0]))
model.fit(X, y, batch_size=32, epochs=165, validation_split=0.1)
