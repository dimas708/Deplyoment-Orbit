import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout, GlobalAveragePooling2D
from keras import activations

def make_model():
    DenseNet = tf.keras.applications.DenseNet201(input_shape=(224 , 224, 3),
                                                    include_top=False,
                                                    weights='imagenet')

    #digabungkan dengan fully connected layer
    model = Sequential()
    model.add(DenseNet)
    model.add(GlobalAveragePooling2D())
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(512, activation="relu"))
    model.add(Dense(9, activation="softmax" , name="classification"))

    return model