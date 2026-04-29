import numpy as np
import tensorflow as tf
from PIL import Image


def load_keras_model(model_path):
    return tf.keras.models.load_model(model_path, compile=False)


def get_input_size(model):
    shape = model.input_shape

    if isinstance(shape, list):
        shape = shape[0]

    height = shape[1] if shape[1] is not None else 224
    width = shape[2] if shape[2] is not None else 224

    return int(width), int(height)


def prepare_image(image: Image.Image, input_size, preprocess_function):
    image = image.convert("RGB")
    image = image.resize(input_size)

    img_array = np.array(image).astype("float32")
    img_array = np.expand_dims(img_array, axis=0)

    img_array = preprocess_function(img_array)

    return img_array