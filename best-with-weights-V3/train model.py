import matplotlib.pyplot as plt
import os
from keras.preprocessing.image import ImageDataGenerator
import seaborn as sns
import tensorflow as tf
import pickle


class_weights = {0: 1, 1: 4}

mainDIR = os.listdir('E:/dataset MIDV HOLO/Mosaics V3 splited final/train')

train_folder = 'E:/dataset MIDV HOLO/Mosaics V3 splited final/train'
val_folder = 'E:/dataset MIDV HOLO/Mosaics V3 splited final/validation'
test_folder = 'E:/dataset MIDV HOLO/Mosaics V3 splited final/test'

train_datagen = ImageDataGenerator(rotation_range=10,
                                   shear_range=20,
                                   horizontal_flip=True,
                                   vertical_flip=True)
val_datagen = ImageDataGenerator()

training_set = train_datagen.flow_from_directory(train_folder,
                                                 target_size=(224, 224),
                                                 batch_size=64,
                                                 class_mode='binary'
                                                 )

validation_set = val_datagen.flow_from_directory(val_folder,
                                                 target_size=(224, 224),
                                                 batch_size=64,
                                                 class_mode='binary'
                                                 )

base_model = tf.keras.applications.MobileNet(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

number_of_layers = len(base_model.layers)
percent = 0.32
number_of_layers_to_freeze = int(number_of_layers * percent)

for layer in base_model.layers[:number_of_layers_to_freeze]:
    layer.trainable = False

model = tf.keras.models.Sequential()
model.add(base_model)
model.add(tf.keras.layers.GlobalAveragePooling2D())
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

model.build((None, 224, 224, 3))

# Define the learning rate
learning_rate = 0.0001

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

model.compile(
    optimizer=optimizer,
    loss='binary_crossentropy',
    metrics=['acc'])

model.summary()

# Define the filepath where you want to save the best model
filepath = "best-with-weights-V3/best_model.h5"

# Define the ModelCheckpoint callback to monitor validation accuracy and save the best model
checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')


history = model.fit(training_set, validation_data=validation_set, epochs=50, batch_size=64,
                    steps_per_epoch=len(training_set),
                    validation_steps=len(validation_set),
                    class_weight=class_weights,
                    callbacks=[checkpoint])

model.save('best-with-weights-V3/last_epoch_model.h5')

with open('best-with-weights-V3/history.pkl', 'wb') as file:
    pickle.dump(history.history, file)

fig, ax = plt.subplots(figsize=(20, 8))
sns.lineplot(x=history.epoch, y=history.history['loss'], marker="o")
sns.lineplot(x=history.epoch, y=history.history['val_loss'], marker="o")
ax.set_title('Learning Curve (Loss)')
ax.set_ylabel('Loss')
ax.set_xlabel('Epoch')
ax.legend(['train', 'val'], loc='best')
plt.savefig('best-with-weights-V3/Learning Curve (Loss).png')
plt.show()

fig, ax = plt.subplots(figsize=(20, 8))
sns.lineplot(x=history.epoch, y=history.history['acc'], marker="o")
sns.lineplot(x=history.epoch, y=history.history['val_acc'], marker="o")
ax.set_title('Learning Curve (Accuracy)')
ax.set_ylabel('Accuracy')
ax.set_xlabel('Epoch')
ax.legend(['train', 'val'], loc='best')
plt.savefig('best-with-weights-V3/Learning Curve (Accuracy).png')
plt.show()
