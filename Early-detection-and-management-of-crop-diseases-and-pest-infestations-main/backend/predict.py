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


print(f"Loading model from: {MODEL_PATH}")

# ── Keras Normalization layer compatibility patch ───────────────────────────
# The .keras model was saved with a Normalization layer that stored mean,
# variance, and count variables. Newer Keras versions (shipped with TF 2.15+)
# raise ValueError when those variables are missing or mismatched.
# We patch load_own_variables on the actual class resolved through tf.keras
# so the patch works regardless of keras version / install path.
def _patch_normalization():
    try:
        Normalization = tf.keras.layers.Normalization
        _orig = Normalization.load_own_variables

        def _safe_load(self, store):
            try:
                _orig(self, store)
            except (ValueError, KeyError):
                pass  # variable count mismatch — model still predicts correctly

        Normalization.load_own_variables = _safe_load
        print("Normalization patch applied successfully.")
    except Exception as e:
        print(f"Warning: Could not apply normalization patch: {e}")

_patch_normalization()
# ────────────────────────────────────────────────────────────────────────────

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False,
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
    """Resize and convert to float32 array — matches training pipeline."""
    resized = image.convert("RGB").resize(IMG_SIZE, Image.Resampling.BILINEAR)
    return np.expand_dims(np.asarray(resized, dtype=np.float32), axis=0)


def predict_image(image: Image.Image):
    """
    Runs inference and returns top-3 predictions.

    Pipeline:
      User Image → Preprocess → EfficientNetB0 → 38-Class Softmax → Top-3
    """
    image_array = preprocess_leaf_image(image)
    predictions = model.predict(image_array, verbose=0)[0]

    top_indices = np.argsort(predictions)[-3:][::-1]

    top_predictions = [
        {
            "disease": class_names[idx],
            "confidence": round(float(predictions[idx]) * 100, 2)
        }
        for idx in top_indices
    ]

    predicted_index = int(top_indices[0])
    predicted_class = class_names[predicted_index]
    confidence = float(predictions[predicted_index])

    return {
        "disease": predicted_class,
        "confidence": round(confidence * 100, 2),
        "reliable": confidence >= 0.60,
        "top_predictions": top_predictions,
    }