import os
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet50_preprocess

BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "model")

WPD_CLASSES = [
    "Black Rust",
    "Blast",
    "Brown Rust",
    "Common Root Rot",
    "Fusarium Head Blight",
    "Healthy",
    "Leaf Blight",
    "Mildew",
    "Septoria",
    "Smut",
    "Tan Spot",
    "Yellow Rust"
]

LWDC_CLASSES = [
    "Healthy Wheat",
    "Leaf Rust",
    "Wheat Loose Smut"
]

def simple_rescale(x):
    return x / 255.0


MODELS = {
    "ResNet50 - WPD": {
        "path": os.path.join(MODEL_DIR, "ResNet50_WPD.keras"),
        "classes": WPD_CLASSES,
        "preprocess": resnet50_preprocess,
        "training accuracy": 0.90,
        "final evaluation accuracy": 0.32
    },
    "ResNet50 - LWDC": {
        "path": os.path.join(MODEL_DIR, "ResNet50_LWDCD.keras"),
        "classes": LWDC_CLASSES,
        "preprocess": resnet50_preprocess,
        "training accuracy": 0.97,
        "final evaluation accuracy": 0.82
    },
    "VGG16 - WPD": {
        "path": os.path.join(MODEL_DIR, "VGG16_WPD.keras"),
        "classes": WPD_CLASSES,
        "preprocess": vgg16_preprocess,
        "training accuracy": 0.86,
        "final evaluation accuracy": 0.38
    },
    "VGG16 - LWDC": {
        "path": os.path.join(MODEL_DIR, "VGG16_LWDCD.keras"),
        "classes": LWDC_CLASSES,
        "preprocess": vgg16_preprocess,
        "training accuracy": 0.95,
        "final evaluation accuracy": 0.80
    },
    "WheatNetwork - WPD": {
        "path": os.path.join(MODEL_DIR, "WheatNetwork_WPD.keras"),
        "classes": WPD_CLASSES,
        "preprocess": simple_rescale,
        "training accuracy": 0.86,
        "final evaluation accuracy": 0.28
    },
    "WheatNetwork - LWDC": {
        "path": os.path.join(MODEL_DIR, "WheatNetwork_LWDCD.keras"),
        "classes": LWDC_CLASSES,
        "preprocess": simple_rescale,
        "training accuracy": 0.85,
        "final evaluation accuracy": 0.71
    }
}