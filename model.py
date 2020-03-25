import os

'''
Try to import PlaidML backend (and ignore errors).
PlaidML enables GPU support via OpenCL, which allows AMD devices to be used.
'''
try:
    import plaidml.keras

    plaidml.keras.install_backend()
    os.environ['KERAS_BACKEND'] = 'plaidml.keras.backend'
except ImportError:
    print("PlaidML not found, using default Keras backend")

from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.utils import shuffle
import cv2
import numpy as np
import sklearn
import glob
import csv
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


'''
Load the training images from Records directory.
All csv files in that directory will be iterated. 
'''
samples = []
for filename in glob.iglob("Records/*.csv"):
    with open(filename) as csvfile:
        reader = csv.reader(csvfile)

        print("reading " + filename)
        for line in reader:
            center_angle = float(line[3])
            center_image = 'Records/IMG/' + line[0].split('/')[-1]
            left_image = 'Records/IMG/' + line[1].split('/')[-1]
            right_image = 'Records/IMG/' + line[2].split('/')[-1]

            samples.append((center_image, center_angle))
            samples.append([left_image, center_angle + 0.2])
            samples.append([right_image, center_angle - 0.2])


print("we have " + str(len(samples)) + " samples")

# Use 20% of the data as validation data
train_samples, validation_samples = train_test_split(samples, test_size=0.2)


'''
Generate batches for training. Samples are shuffled.
Each sample image is converted to YUV to disentangle luma and chroma data.
Each image is also flipped to help better generalize the model
'''
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1:  # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                image = cv2.imread(batch_sample[0])
                if image is None:
                    print("Failed to read image " + batch_sample)
                else:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
                    angles.append(batch_sample[1])

                    images.append(np.fliplr(image))
                    angles.append(-batch_sample[1])

            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)


## MODEL BEGINS HERE


from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Conv2D, Dropout, LSTM, TimeDistributed

batch_size = 150
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)


'''
Trainig model based on NVidia CNN network. 5 Convolutions with 3 fully connected layers.
Data on input is normalized and cropped to ROI.
Convolutional layers use Relu as activation to introduce nonlinearity.
To prevent overfitting Dropout of 50% is included before each Dense layer. 
'''
model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3), output_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((50, 20), (0, 0))))
model.add(Conv2D(24, 5, strides=(2, 2), activation='relu'))
model.add(Conv2D(36, 5, strides=(2, 2), activation='relu'))
model.add(Conv2D(48, 5, strides=(2, 2), activation='relu'))
model.add(Conv2D(64, 3, activation='relu'))
model.add(Conv2D(64, 3, activation='relu'))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(50))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Dropout(0.5))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

'''
Define early stopping so that we won't continue bad experiments. Also - save the best model so far (in case of power outage ;) ).
'''
callbacks = [EarlyStopping(monitor='val_loss', patience=3, verbose=1),
             ModelCheckpoint('model_chk.h5', monitor='val_loss', save_best_only=True, verbose=0)]

# Train model using generator
history_object = model.fit_generator(train_generator,
                                     steps_per_epoch=np.ceil(len(train_samples) / batch_size),
                                     validation_data=validation_generator,
                                     validation_steps=np.ceil(len(validation_samples) / batch_size),
                                     epochs=10,
                                     verbose=1,
                                     callbacks=callbacks)

# Save trained model
model.save('model.h5')


# Plot how good the model is. Does not directly mean good driving, as it may still fail on hard cases causing car to crash.
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
