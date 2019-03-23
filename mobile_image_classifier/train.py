#!/usr/bin/env python
# coding: utf-8

# In[10]:


# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")
 
# import the necessary packages
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import pickle
import os
import pandas as pd
import cv2
from PIL import Image
from keras.callbacks import ModelCheckpoint
from math import ceil

from smallervggnet import SmallerVGGNet

# initialize the number of epochs to train for, initial learning rate,
# batch size, and image dimensions

EPOCHS = 30
INIT_LR = 1e-3
BS = 128
IMAGE_DIMS = (96, 96, 3)

def train(dataset):
    dataset.Category = dataset.Category.apply(str)
    
    unique_categories = dataset['Category'].unique().tolist()
    
    # binarize the labels using scikit-learn's special multi-label
    # binarizer implementation
    print("[INFO] class labels: " + str(unique_categories))
    
    lb = LabelBinarizer()
    labels = lb.fit_transform(unique_categories)

    trainset, testset = train_test_split(dataset, test_size=0.2, random_state=42)
    
    # construct the image generator for data augmentation
    image_data_gen = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
        height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
        horizontal_flip=True, fill_mode="nearest", rescale=1/255.0, dtype=float)
    
    trainset_iter = image_data_gen.flow_from_dataframe(trainset,
                                                        directory='./',
                                                        x_col='image_path',
                                                        y_col='Category',
                                                        target_size=(IMAGE_DIMS[1], IMAGE_DIMS[0]),
                                                        color_mode='rgb',
                                                        class_mode='categorical',
                                                        classes=unique_categories,
                                                        batch_size=BS)
    
    testset_iter = image_data_gen.flow_from_dataframe(testset,
                                                    directory='./',
                                                    x_col='image_path',
                                                    y_col='Category',
                                                    target_size=(IMAGE_DIMS[1], IMAGE_DIMS[0]),
                                                    color_mode='rgb',
                                                    class_mode='categorical',
                                                    classes=unique_categories,
                                                    batch_size=BS)
    
    # initialize the model using a sigmoid activation as the final layer
    # in the network so we can perform multi-label classification
    print("[INFO] compiling model...")
    model = SmallerVGGNet.build(
        width=IMAGE_DIMS[1], height=IMAGE_DIMS[0],
        depth=IMAGE_DIMS[2], classes=len(unique_categories),
        finalAct="softmax")
    # initialize the optimizer
    opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
    
    # compile the model using binary cross-entropy rather than
    # categorical cross-entropy -- this may seem counterintuitive for
    # multi-label classification, but keep in mind that the goal here
    # is to treat each output label as an independent Bernoulli
    # distribution
    model.compile(loss="binary_crossentropy", optimizer=opt,
        metrics=["accuracy"])
    
    # checkpoint
    filepath="weights-improvement-{epoch:02d}-{val_acc:.2f}.model"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    # train the network
    print("[INFO] training network...")
    H = model.fit_generator(
        trainset_iter,
        validation_data=testset_iter,
        steps_per_epoch=ceil(len(trainset) / BS),
        validation_steps=ceil(len(testset) / BS),
        epochs=EPOCHS, verbose=1, use_multiprocessing=True, callbacks=callbacks_list)
    
    return (model, lb, H)

def is_interactive():
    import __main__ as main
    return not hasattr(main, '__file__')


# ### Set the arguments, if using a notebook then edit in `else`.

# In[11]:


# construct the argument parse and parse the arguments
if not is_interactive():
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", required=True,
        help="path to train.csv")
    ap.add_argument("-m", "--model", required=True,
        help="path to output model")
    ap.add_argument("-l", "--labelbin", required=True,
        help="path to output label binarizer")
    ap.add_argument("-p", "--plot", type=str, default="plot.png",
        help="path to output accuracy/loss plot")
    args = vars(ap.parse_args())
else:
    args = {}
    args['dataset'] = '../train.csv'
    args['model'] = 'mobile_categorizer.model'
    args['labelbin'] = 'lb.pickle'
    args["plot"] = 'plot.png'


# ### Runs the training

# In[ ]:


dataset = pd.read_csv(args['dataset'])

dataset = dataset[dataset
                      .image_path
                      .apply(lambda image_path: image_path.startswith('mobile_image'))]

dataset = dataset.head(int(len(dataset)/1))

model, lb, H = train(dataset)


# In[ ]:


# save the model to disk
print("[INFO] serializing network...")
model.save(args["model"])

# save the label binarizer to disk
print("[INFO] serializing label binarizer...")
f = open(args["labelbin"], "wb")
f.write(pickle.dumps(lb))
f.close()

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
N = EPOCHS
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper left")
plt.savefig(args["plot"])


# In[ ]:




