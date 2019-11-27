import numpy as np
import torch
from sklearn import datasets
from matplotlib import pyplot as plt
import torch
import torchvision
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms

from matplotlib import image
from matplotlib import pyplot
from PIL import Image
from tensorflow.keras.datasets import mnist
from tensorflow.keras import backend
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot
import os
import cv2
import glob
import numpy
import PIL
image_list = []
for filename in glob.glob('compressed/*.gif'):
    img = PIL.Image.open(filename).convert("L")
    imgarr = numpy.array(img)
    image_list.append(imgarr)

print(image_list)
print(image_list[0])
image_list=np.asarray(image_list)
X_train = image_list.reshape((image_list.shape[0], 30, 30, 1))
X_train = X_train.astype('float32')
def aug_type(x):
    if x == "normalization":
        datagen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True)
        dir = "C:/Users/zhaoj/PycharmProjects/421HW5/normal"
    elif x == "zca_whitening":
        datagen = ImageDataGenerator(zca_whitening=True)
        dir = "C:/Users/zhaoj/PycharmProjects/421HW5/whitening"
    elif x == "rotation":
        datagen = ImageDataGenerator(rotation_range=90)
        dir = "C:/Users/zhaoj/PycharmProjects/421HW5/rotation"
    elif x == "shift":
        shift = 0.2
        datagen = ImageDataGenerator(width_shift_range=shift, height_shift_range=shift)
        dir = "C:/Users/zhaoj/PycharmProjects/421HW5/shift"
    elif x == "flip":
        datagen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True)
        dir = "C:/Users/zhaoj/PycharmProjects/421HW5/flip"
    print("here")
    datagen.fit(X_train)
    try:
        os.mkdir(dir)
    except OSError:
        print("Creation of the directory %s failed" % dir)
    else:
        print("Successfully created the directory %s " % dir)
    f = datagen.flow(X_train,batch_size=9,save_to_dir=dir,save_format='gif')
    count = 0
    print("f lenth")
    print(len(f))
    for X_batch in f:

        count = count + 1
        if count ==19:
            break
        for i in range(0, 9):
            pyplot.subplot(330 + 1 + i)
            pyplot.imshow(X_batch[i].reshape(30, 30), cmap=pyplot.get_cmap('gray'))


    print(count)
#aug_type("flip")
#aug_type("shift")
#aug_type("rotation")
#aug_type("zca_whitening")
#aug_type("normalization")