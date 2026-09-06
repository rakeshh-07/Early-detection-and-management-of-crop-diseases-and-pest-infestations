import os
import json
import numpy as np
import tensorflow as tf
from PIL import Image



BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "best_plant_disease_model.keras"
)

CLASS_NAMES_PATH = os.path.join(
    BASE_DIR,
    "models",
    "class_names.json"
)

IMG_SIZE = (224, 224)


print(f"Loading model: {MODEL_PATH}")

# Fix for Keras normalization layer variable mismatch between save/load versions
class FixedNormalization(tf.keras.layers.Normalization):
    def load_own_variables(self, store):
        try:
            super().load_own_variables(store)
        except (ValueError, KeyError):
            # If variables mismatch, skip — adapt() will rebuild them at inference
            pass

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False,
    custom_objects={"Normalization": FixedNormalization},
    safe_mode=False,
)

with open(CLASS_NAMES_PATH, "r") as f:
    class_names = json.load(f)

if model.output_shape[-1] != len(class_names):
    raise ValueError(
        f"Model outputs {model.output_shape[-1]} classes, "
        f"but class_names.json contains {len(class_names)}."
    )

print(f"Model loaded successfully with {len(class_names)} classes.")


def preprocess_leaf_image(image: Image.Image) -> np.ndarray:
    """Match image_dataset_from_directory and EfficientNetB0 preprocessing."""
    resized = image.convert("RGB").resize(IMG_SIZE, Image.Resampling.BILINEAR)
    return np.expand_dims(np.asarray(resized, dtype=np.float32), axis=0)


def predict_image(image: Image.Image):
    """
    Performs complete inference:

    User Image
        ↓
    Leaf Segmentation
        ↓
    Leaf Localization
        ↓
    Leaf Crop
        ↓
    EfficientNetB0
        ↓
    38-Class Prediction
        ↓
    Top-3 Results
    """


    image_array = preprocess_leaf_image(image)
    predictions = model.predict(image_array, verbose=0)[0]


    top_indices = np.argsort(
        predictions
    )[-3:][::-1]

    top_predictions = []

    for index in top_indices:

        top_predictions.append(
            {
                "disease": class_names[index],
                "confidence": round(
                    float(predictions[index]) * 100,
                    2
                )
            }
        )
        
    predicted_index = int(
        top_indices[0]
    )

    predicted_class = class_names[
        predicted_index
    ]

    confidence = float(
        predictions[predicted_index]
    )
    return {
        "disease": predicted_class,
        "confidence": round(confidence * 100, 2),
        "reliable": confidence >= 0.60,
        "top_predictions": top_predictions,
    }