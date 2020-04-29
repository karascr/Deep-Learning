import os
import numpy as np
import glob
import shutil
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator

"""
URL = 'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz'

zip_file = tf.keras.utils.get_file(origin=URL, fname="flower_photos.tgz", extract=True)

base_dir = os.path.join(os.path.dirname(zip_file), 'flower_photos')
"""
base_dir = 'C:\\Users\kara\.keras\datasets\\flower_photos'

"""
classes = ['roses', 'daisy', 'dandelion', 'sunflowers', 'tulips']

for cl in classes:
    cl_path = os.path.join(base_dir, cl)
    images = glob.glob(cl_path + '/*.jpg')
    print('{}: {} images'.format(cl, len(images)))
    num_train = int(round( len(images) * 0.8 ))
    train, val = images[:num_train], images[num_train:]

    cl_train_path = os.path.join(base_dir, 'train', cl)
    for t in train:
        if not os.path.exists(cl_train_path):
            os.makedirs(cl_train_path)
        shutil.move(t, cl_train_path)

    cl_val_path = os.path.join(base_dir, 'val', cl)
    for v in val:
        if not os.path.exists(cl_val_path):
            os.makedirs(cl_val_path)
        shutil.move(v, cl_val_path)

train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')
"""

### DATA AUGMENTATION ###

batch_size = 100
IMG_SHAPE = 150

"""
    rescales the images by 255 
    random 45 degree rotation
    random zoom of up to 50%
    random horizontal flip
    width shift of 0.15
    height shift of 0.15

"""

image_gen = ImageDataGenerator(rescale=1./255,
                               rotation_range=45,
                               width_shift_range=.15,
                               height_shift_range=.15,
                               horizontal_flip=True,
                               zoom_range=.5
                               )

train_data_gen = image_gen.flow_from_directory(train_dir,
                                               (IMG_SHAPE, IMG_SHAPE),
                                               batch_size=batch_size,
                                               shuffle=True,
                                               class_mode='sparse'
                                               )

image_gen_val = ImageDataGenerator(rescale=1./255)

val_data_gen = image_gen_val.flow_from_directory(val_dir,
                                                 (IMG_SHAPE, IMG_SHAPE),
                                                 batch_size=batch_size,
                                                 class_mode='sparse'
                                                 )

### CNN MODEL ###
"""
In the cell below, create a convolutional neural network that consists of 3 convolution 
blocks. Each convolutional block contains a Conv2D layer followed by a max pool layer. 
The first convolutional block should have 16 filters, the second one should have 32 filters,
and the third one should have 64 filters. All convolutional filters should be 3 x 3. 
All max pool layers should have a pool_size of (2, 2).

After the 3 convolutional blocks you should have a flatten layer followed by a fully connected
layer with 512 units. The CNN should output class probabilities based on 5 classes which is 
done by the softmax activation function. All other layers should use a relu activation 
function. You should also add Dropout layers with a probability of 20%, where appropriate
"""

model = Sequential([
    Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_SHAPE, IMG_SHAPE, 3)),
    MaxPooling2D((2, 2)),

    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D((2, 2)),

    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D((2, 2)),

    Flatten(),
    Dropout(0.2),
    Dense(512, activation='relu'),
    Dropout(0.2),
    Dense(5)
])

model.compile('adam',
              tf.keras.losses.SparseCategoricalCrossentropy(True),
              ['accuracy'])

epochs = 80

history = model.fit_generator(generator=train_data_gen,
                              steps_per_epoch=int(np.ceil(train_data_gen.n / float(batch_size))),
                              epochs=epochs,
                              validation_data=val_data_gen,
                              validation_steps=int(np.ceil(val_data_gen.n / float(batch_size))))

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))

plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')

plt.show()