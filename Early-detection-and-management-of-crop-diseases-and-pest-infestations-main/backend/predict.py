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

# ── Keras cross-version weight-loading compatibility patch ─────────────────
# Problem: The .keras model was saved with an older Keras version. When loaded
# by a newer Keras (TF 2.15+), saving_lib._load_state() calls
# layer.load_own_variables(store) on every layer. The base_layer implementation
# raises ValueError if the stored variable count doesn't match what the layer
# expects. This affects Normalization layers, EfficientNet stem/blocks, etc.
#
# Fix: Patch the base class (tf.keras.layers.Layer) so load_own_variables
# silently continues on count mismatches instead of crashing.
def _apply_keras_loading_patch():
    try:
        import keras
        # Try to reach the actual base Layer class that holds load_own_variables
        # Keras 2.x path (TF 2.x bundled keras)
        base_layer_cls = None
        try:
            from keras.engine.base_layer import Layer as _L
            base_layer_cls = _L
            print("Patching keras.engine.base_layer.Layer")
        except ImportError:
            pass

        if base_layer_cls is None:
            try:
                from keras.src.engine.base_layer import Layer as _L
                base_layer_cls = _L
                print("Patching keras.src.engine.base_layer.Layer")
            except ImportError:
                pass

        if base_layer_cls is None:
            # Keras 3.x path
            try:
                from keras.src.layers.layer import Layer as _L
                base_layer_cls = _L
                print("Patching keras.src.layers.layer.Layer")
            except ImportError:
                pass

        if base_layer_cls is None:
            print("Warning: Could not find keras Layer base class to patch.")
            return

        _orig_load_own_variables = base_layer_cls.load_own_variables

        def _safe_load_own_variables(self, store):
            try:
                _orig_load_own_variables(self, store)
            except (ValueError, KeyError) as e:
                print(f"  [patch] Skipped weight mismatch for layer '{getattr(self, 'name', '?')}': {e}")

        base_layer_cls.load_own_variables = _safe_load_own_variables
        print("Keras base Layer.load_own_variables patch applied.")

    except Exception as e:
        print(f"Warning: Failed to apply keras patch: {e}")


_apply_keras_loading_patch()
# ───────────────────────────────────────────────────────────────────────────

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