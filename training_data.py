# -*- coding: utf-8 -*-
"""
Created on Sun Dec 27 14:59:30 2020

@author: Rupak Kumar Das
"""

from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.applications.mobilenet import preprocess_input
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")

#------------------------------------------------------------------------------
# Data Preprocessing
DIRECTORY = "dataset"
CATEGORIES = ["with mask", "without mask"]

epochs_value=1
BS=32
data = []
target = []

for category in CATEGORIES:
    path = os.path.join(DIRECTORY, category)
    for img in os.listdir(path):
       	img_path = os.path.join(path, img)
       	image = load_img(img_path, target_size=(224, 224, 3), color_mode='rgb')
       	image = img_to_array(image)
        image = preprocess_input(image)    
        data.append(image)
       	target.append(category)

## labeling the target values
lb = LabelBinarizer() 
target = lb.fit_transform(target)
target = to_categorical(target)

data = np.array(data, dtype="float32")
target = np.array(target)

#------------------------------------------------------------------------------
#Training and Model design 
from sklearn.model_selection import train_test_split
(train_data,test_data,train_target,test_target)=train_test_split(data,target,test_size=0.2, stratify=target, random_state=42)


from keras_preprocessing.image import ImageDataGenerator

aug = ImageDataGenerator(
 	rotation_range=20,
 	zoom_range=0.20,
 	width_shift_range=0.2,
 	height_shift_range=0.2,
 	shear_range=0.15,
 	horizontal_flip=True,
 	fill_mode="nearest")


from keras.models import Sequential
from keras.layers import Dense,Activation,Flatten,Dropout
from keras.layers import Conv2D,MaxPooling2D


from tensorflow.keras.optimizers import SGD
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(224, 224, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(2, activation='sigmoid'))
# compile model
opt = SGD(lr=0.001, momentum=0.9)
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])


H=model.fit(
    aug.flow(train_data,train_target,batch_size=BS),
    steps_per_epoch=len(train_data)//BS,
    validation_data=(test_data,test_target),
    validation_steps=len(test_data)//BS,
    epochs=epochs_value)

#------------------------------------------------------------------------------
#Predict the output
prediction = model.predict(test_data, batch_size=BS)
prediction = np.argmax(prediction, axis=1)



from sklearn.metrics import classification_report
print(classification_report(test_target.argmax(axis=1), prediction,
	target_names=lb.classes_))


model.save("Face_mask_detector.model", save_format="h5")

import matplotlib.pyplot as plt
N = epochs_value
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="training loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="validation loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="training accuracy")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="validation accuracy")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")

