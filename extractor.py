import os
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import optimizers
import numpy as np
from smallcnn import save_history
import tensorflow as tf

config = tf.ConfigProto(gpu_options=tf.GPUOptions(visible_device_list="0,1"))
tf.Session(config=config)

img_width, img_height = 150, 150
train_data_dir = 'imageData/train'
validation_data_dir = 'imageData/validation'
nb_train_samples = 660
nb_validation_samples = 300
nb_epoch = 1000

result_dir = 'results'
if not os.path.exists(result_dir):
    os.mkdir(result_dir)


def save_bottleneck_features():
    model = VGG16(include_top=False, weights='imagenet')
    model.summary()

    datagen = ImageDataGenerator(rescale=1.0 / 255)
    generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=1,
        class_mode=None,
        shuffle=False)

    bottleneck_features_train = model.predict_generator(generator, nb_train_samples)
    np.save(os.path.join(result_dir, 'bottleneck_features_train.npy'),
            bottleneck_features_train)
    generator = datagen.flow_from_directory(
                validation_data_dir,
                target_size=(img_width, img_height),
                batch_size=1,
                class_mode=None,
                shuffle=False)

    bottleneck_features_validation = model.predict_generator(generator, nb_validation_samples)
    np.save(os.path.join(result_dir, 'bottleneck_features_validation.npy'),
            bottleneck_features_validation)

def train_top_model():
    train_data = np.load(os.path.join(result_dir, 'bottleneck_features_train.npy'))
    train_labels = np.array([0] * int(nb_train_samples / 2) + [1] * int(nb_train_samples / 2))
    print(train_data.shape)

    validation_data = np.load(os.path.join(result_dir, 'bottleneck_features_validation.npy'))
    validation_labels = np.array([0] * int(nb_validation_samples / 2) + [1] * int(nb_validation_samples / 2))

    print(validation_data.shape)

    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
                  metrics=['accuracy'])

    history = model.fit(train_data, train_labels,
                        nb_epoch=nb_epoch,
                        batch_size=32,
                        validation_data=(validation_data, validation_labels))
    model.save_weights(os.path.join(result_dir, 'bottleneck_fc_model.h5'))
    save_history(history, os.path.join(result_dir, 'history_extractor.txt'))

if __name__ == '__main__':
    save_bottleneck_features()
    train_top_model()
            
