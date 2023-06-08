from keras.models import load_model, Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array

from PIL import Image
import numpy as np
import os

img_width, img_heigh = 150, 150
trainRescale = ImageDataGenerator(rescale=1./255)
trainData = trainRescale.flow_from_directory(
    'train/',
    target_size=(img_heigh, img_width),
    batch_size=32,
    class_mode="binary")

model = Sequential()

model.add(Conv2D(32,(3,3), input_shape =(img_width, img_heigh, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(
    loss='binary_crossentropy',
    optimizer = 'rmsprop',
    metrics = ['accuracy'])

model.fit(
    trainData,
    epochs=21,
    steps_per_epoch=1,
)


model.save_weights('models_weights.h5')
model.save('model_keras.h5')

test_imgs = os.listdir('test/')

for image in test_imgs:
    img = Image.open('test/'+ image)
    img = img.resize((img_width, img_heigh))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)

    result = model.predict(img)
    if result[0][0] >= 0.5:
        predicition = 'com peste'
    else:
        predicition = 'normal'

    print("a img", image, "é um plantação: ", predicition)