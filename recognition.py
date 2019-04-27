import numpy as np
import matplotlib.pyplot as plt
import os
from shutil import copy

from keras.layers import Convolution2D, MaxPooling2D, Flatten
from keras.layers import Dense
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator


def load_make_dir(path, i):
    file_org = 'train_signatures/org/'
    file_forg = 'train_signatures/forg/'
    path_ = 'train_signatures'
    for directory in os.listdir(path):
        if not directory.endswith('txt'):
           for file in os.listdir(path+'/'+directory):
               if file.endswith('png'):
                   file_name = file.split('_')
                   if int(file_name[1]) == i:
                       if file_name[0] == 'original':
                           print(file)
                           #image_org_dir.append(path+'/'+directory+'/'+file)
                           src = path+'/'+directory+'/'+file
                           copy(src, file_org)
                       else:
                           print(file)
                           #image_forg_dir.append(path+'/'+directory+'/'+file)
                           src = path+'/'+directory+'/'+file
                           copy(src, file_forg)

    return path_

def delete_file(path):
  for directory in os.listdir(path):
    for file in os.listdir(path+'/'+directory):
        os.unlink(path+'/'+directory+'/'+file)

i = 1
file_path = load_make_dir('signatures', i)

classifier = Sequential()

# Step 1 - Convolution
classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(output_dim = 128, activation = 'relu'))
classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory(file_path,
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

classifier.fit_generator(training_set,
                         samples_per_epoch = 8000,
                         nb_epoch = 5,
                         #validation_data = test_set,
                         #nb_val_samples = 2000,
                         validation_steps=2)
try:
    delete_file(str(file_path))
except:
    delete_file('train_signatures')
        
# serialize model to json
model_json = classifier.to_json()
save_model = 'model_save/'
with open(save_model+'model{}.json'.format(i), 'w') as json_file:
    json_file.write(model_json)
# serailize weights to HDF5
classifier.save_weights(save_model+'model{}.h5'.format(i))
print('Saved model to the disk')
