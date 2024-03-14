import matplotlib.pyplot as plt
import os
from keras.preprocessing.image import ImageDataGenerator
import seaborn as sns
import tensorflow as tf
import pickle

# model = tf.keras.applications.MobileNet()
"""# **Data**"""

mainDIR = os.listdir('E:/dataset MIDV HOLO/Mosaics splited/train')

train_folder = 'E:/dataset MIDV HOLO/Mosaics splited/train'
val_folder = 'E:/dataset MIDV HOLO/Mosaics splited/val'
test_folder = 'E:/dataset MIDV HOLO/Mosaics splited/test'
"""

"""  # **Model**"""

# data gens
train_datagen = ImageDataGenerator(rotation_range=10,
                                   shear_range=20,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   vertical_flip=True
                                   )

val_datagen = ImageDataGenerator()  # Image normalization.

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

# Load the pre-trained MobileNet model without the top layer
base_model = tf.keras.applications.MobileNet(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

number_of_layers = len(base_model.layers)

# Freeze the % layers in the base model
percent = 0.32
number_of_layers_to_freeze = int(number_of_layers * percent)
# Freeze some layers
for layer in base_model.layers[:number_of_layers_to_freeze]:
    layer.trainable = False

# description of the model
model = tf.keras.models.Sequential()
model.add(base_model)
model.add(tf.keras.layers.GlobalAveragePooling2D())
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

# Build the model
model.build((None, 224, 224, 3))

# Compile the model
model.compile(
    optimizer="adamax",
    loss='binary_crossentropy',
    metrics=['acc'])

model.summary()
# callbacks


# train

history = model.fit(training_set, validation_data=validation_set, epochs=50, batch_size=64,
                    steps_per_epoch=len(training_set),
                    validation_steps=len(validation_set)
                    )

# sauvgarder le model
model.save('best-without-weights/last_epoch_model.h5')
# save history
with open('best-without-weights/history.pkl', 'wb') as file:
    pickle.dump(history.history, file)

# Learning Curve (Loss)
fig, ax = plt.subplots(figsize=(20, 8))
sns.lineplot(x=history.epoch, y=history.history['loss'], marker="o")
sns.lineplot(x=history.epoch, y=history.history['val_loss'], marker="o")
ax.set_title('Learning Curve (Loss)')
ax.set_ylabel('Loss')
ax.set_xlabel('Epoch')
ax.legend(['train', 'val'], loc='best')
plt.savefig('best-without-weights/Learning Curve (Loss).png')
plt.show()

"""*   Learning Curve (Accuracy)"""

# Learning Curve (Accuracy)
fig, ax = plt.subplots(figsize=(20, 8))
sns.lineplot(x=history.epoch, y=history.history['acc'], marker="o")
sns.lineplot(x=history.epoch, y=history.history['val_acc'], marker="o")
ax.set_title('Learning Curve (Accuracy)')
ax.set_ylabel('Accuracy')
ax.set_xlabel('Epoch')
ax.legend(['train', 'val'], loc='best')
plt.savefig('best-without-weights/Learning Curve (Accuracy).png')
plt.show()
