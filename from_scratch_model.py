import matplotlib.pyplot as plt
import os
from keras.preprocessing.image import ImageDataGenerator
import seaborn as sns
import tensorflow as tf
import pickle
import numpy as np
import cv2

class_weights = {0: 1, 1: 2}

mainDIR = os.listdir('E:/dataset MIDV HOLO/Mosaics V3 splited final/train')

train_folder = 'E:/dataset MIDV HOLO/Mosaics V3 splited final/train'
val_folder = 'E:/dataset MIDV HOLO/Mosaics V3 splited final/validation'
test_folder = 'E:/dataset MIDV HOLO/Mosaics V3 splited final/test'

import matplotlib.pyplot as plt


def preprocess_image(image):
    image = image / 255.0  # Normalize pixel values
    patches = []
    patch_size = (80, 71)
    for i in range(9):
        for j in range(9):
            patch = image[i * patch_size[0]:(i + 1) * patch_size[0], j * patch_size[1]:(j + 1) * patch_size[1]]
            patch = cv2.resize(patch, patch_size)  # Resize patch to fixed size
            patches.append(patch.flatten())
    patches = np.array(patches)
    patches = np.expand_dims(patches, axis=0)  # Add batch dimension
    return patches


train_datagen = ImageDataGenerator(rescale=1. / 255, rotation_range=10,
                                   shear_range=20,
                                   horizontal_flip=True,
                                   vertical_flip=True)
val_datagen = ImageDataGenerator(rescale=1. / 255)

training_set = train_datagen.flow_from_directory(train_folder,
                                                 batch_size=64,
                                                 class_mode='binary',
                                                 target_size=(720,639),
                                                 interpolation='nearest'
                                                 )

validation_set = val_datagen.flow_from_directory(val_folder,
                                                 batch_size=64,
                                                 class_mode='binary',
                                                 target_size=(720,639),
                                                 interpolation='nearest'
                                                 )


# Create Vision Transformer model

class ClassToken(tf.keras.layers.Layer):
    def __init__(self):
        super(ClassToken, self).__init__()

    def build(self, input_shape):
        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(
            initial_value=w_init(shape=(1, 1, input_shape[-1]), dtype=tf.float32),
            trainable=True
        )

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        hidden_dim = self.w.shape[-1]

        cls = tf.broadcast_to(self.w, [batch_size, 1, hidden_dim])
        cls = tf.cast(cls, dtype=inputs.dtype)
        return cls


def mlp(x, cf):
    x = tf.keras.layers.Dense(cf["mlp_dim"], activation="gelu")(x)
    x = tf.keras.layers.Dropout(cf["dropout_rate"])(x)
    x = tf.keras.layers.Dense(cf["hidden_dim"])(x)
    x = tf.keras.layers.Dropout(cf["dropout_rate"])(x)
    return x


def transformer_encoder(x, cf):
    skip_1 = x
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.MultiHeadAttention(
        num_heads=cf["num_heads"], key_dim=cf["hidden_dim"]
    )(x, x)
    x = tf.keras.layers.Add()([x, skip_1])

    skip_2 = x
    x = tf.keras.layers.LayerNormalization()(x)
    x = mlp(x, cf)
    x = tf.keras.layers.Add()([x, skip_2])

    return x


def ViT(cf):
    """ Inputs """
    input_shape = (cf["patch_height"] * 9, cf["patch_width"] * 9, cf["num_channels"])
    inputs = tf.keras.layers.Input(input_shape)

    """ Reshape input to patches """
    patch_height, patch_width = cf["patch_height"], cf["patch_width"]
    num_patches = (input_shape[0] // patch_height) * (input_shape[1] // patch_width)
    patches = tf.image.extract_patches(inputs, sizes=[1, patch_height, patch_width, 1],
                                       strides=[1, patch_height, patch_width, 1], rates=[1, 1, 1, 1], padding='VALID')
    patches = tf.keras.layers.Reshape((num_patches, patch_height * patch_width * cf["num_channels"]))(patches)

    """ Transformer Encoder """
    for _ in range(cf["num_layers"]):
        x = transformer_encoder(patches, cf)

    """ Classification Head """
    x = tf.keras.layers.LayerNormalization()(x)
    x = x[:, 0, :]
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.Dense(cf["num_classes"], activation="softmax")(x)

    model = tf.keras.models.Model(inputs, x)
    return model


config = {}
config["num_layers"] = 12
config["hidden_dim"] = 768
config["mlp_dim"] = 3072
config["num_heads"] = 12
config["dropout_rate"] = 0.1
config["num_patches"] = 9 * 9  # Number of patches
config["patch_width"] = 80  # Patch height
config["patch_height"] = 71  # Patch width
config["num_channels"] = 3
config["num_classes"] = 1  # Number of output classes

vit_model = ViT(config)
vit_model.summary()

learning_rate = 0.0001
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

vit_model.compile(
    optimizer=optimizer,
    loss='binary_crossentropy',
    metrics=['acc'])

history = vit_model.fit(training_set, validation_data=validation_set, epochs=10, batch_size=64,
                        steps_per_epoch=training_set.samples // 64,
                        validation_steps=validation_set.samples // 64,
                        class_weight=class_weights)

vit_model.save('best/last_epoch_model.h5')

with open('best/history.pkl', 'wb') as file:
    pickle.dump(history.history, file)

fig, ax = plt.subplots(figsize=(20, 8))
sns.lineplot(x=history.epoch, y=history.history['loss'], marker="o")
sns.lineplot(x=history.epoch, y=history.history['val_loss'], marker="o")
ax.set_title('Learning Curve (Loss)')
ax.set_ylabel('Loss')
ax.set_xlabel('Epoch')
ax.legend(['train', 'val'], loc='best')
plt.savefig('best/Learning Curve (Loss).png')
plt.show()

fig, ax = plt.subplots(figsize=(20, 8))
sns.lineplot(x=history.epoch, y=history.history['acc'], marker="o")
sns.lineplot(x=history.epoch, y=history.history['val_acc'], marker="o")
ax.set_title('Learning Curve (Accuracy)')
ax.set_ylabel('Accuracy')
ax.set_xlabel('Epoch')
ax.legend(['train', 'val'], loc='best')
plt.savefig('best/Learning Curve (Accuracy).png')
plt.show()
