import os
import json
from pathlib import Path
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorflow.keras import mixed_precision
from sklearn.utils.class_weight import compute_class_weight

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU Memory Growth Enabled")
    except RuntimeError as e:
        print(e)

if gpus:
    mixed_precision.set_global_policy("mixed_float16")
else:
    mixed_precision.set_global_policy("float32")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get(
    "CROP_DATA_DIR",
    os.path.join(BASE_DIR, "Segmented_Dataset")
)
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VALID_DIR = os.path.join(DATA_DIR, "valid")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

IMG_SIZE = (224, 224)
BATCH_SIZE = int(os.environ.get("CROP_BATCH_SIZE", "64"))
SEED = 42
MAX_TRAIN_BATCHES = int(os.environ.get("CROP_MAX_TRAIN_BATCHES", "0"))
MAX_VALID_BATCHES = int(os.environ.get("CROP_MAX_VALID_BATCHES", "0"))
PHASE1_EPOCHS = int(os.environ.get("CROP_PHASE1_EPOCHS", "5"))
PHASE2_EPOCHS = int(os.environ.get("CROP_PHASE2_EPOCHS", "10"))

tf.random.set_seed(SEED)
np.random.seed(SEED)

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED
)

valid_ds = tf.keras.utils.image_dataset_from_directory(
    VALID_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = train_ds.class_names
num_classes = len(class_names)
if valid_ds.class_names != class_names:
    raise ValueError("Training and validation class mappings do not match.")

print(f"Number of classes: {num_classes}")
print("Class mapping:")
for index, class_name in enumerate(class_names):
    print(f"{index} -> {class_name}")

print(f"Training images: {len(train_ds.file_paths)}")
print(f"Validation images: {len(valid_ds.file_paths)}")
print("Training images per class:")
for class_name in class_names:
    count = sum(Path(path).parent.name == class_name for path in train_ds.file_paths)
    print(f"  {class_name}: {count}")

with open(os.path.join(MODEL_DIR, "class_names.json"), "w") as f:
    json.dump(class_names, f, indent=4)

train_labels = [train_ds.class_names.index(Path(p).parent.name) for p in train_ds.file_paths]
class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_labels),
    y=train_labels
)
class_weight_dict = dict(enumerate(class_weights_array))

if MAX_TRAIN_BATCHES > 0:
    train_ds = train_ds.take(MAX_TRAIN_BATCHES)
if MAX_VALID_BATCHES > 0:
    valid_ds = valid_ds.take(MAX_VALID_BATCHES)


AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
valid_ds = valid_ds.prefetch(AUTOTUNE)

data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.15),
        tf.keras.layers.RandomZoom(0.15),
        tf.keras.layers.RandomTranslation(0.1, 0.1),
        tf.keras.layers.RandomContrast(0.2),
    ],
    name="data_augmentation"
)

base_model = tf.keras.applications.EfficientNetB0(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)
base_model.trainable = False

inputs = tf.keras.Input(shape=(224, 224, 3), name="leaf_image")
x = data_augmentation(inputs)
x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.3)(x)

outputs = tf.keras.layers.Dense(
    num_classes,
    activation="softmax",
    dtype="float32",
    name="disease_prediction"
)(x)

model = tf.keras.Model(inputs, outputs, name="PlantDiseaseEfficientNet")

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=4,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        patience=2,
        min_lr=1e-7
    ),
    tf.keras.callbacks.ModelCheckpoint(
        os.path.join(MODEL_DIR, "best_plant_disease_model.weights.h5"),
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=True,
        mode="max"
    )
]

histories = []

print("\nStarting Phase 1: Feature Extraction...")
phase1_history = model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=PHASE1_EPOCHS,
    class_weight=class_weight_dict,
    callbacks=callbacks
)
histories.append(phase1_history.history)

model.load_weights(os.path.join(MODEL_DIR, "best_plant_disease_model.weights.h5"))

base_model.trainable = True
for layer in base_model.layers[:-75]:  
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nStarting Phase 2: Fine-Tuning...")
phase2_history = model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=PHASE2_EPOCHS,
    class_weight=class_weight_dict,
    callbacks=callbacks
)
histories.append(phase2_history.history)

model.load_weights(os.path.join(MODEL_DIR, "best_plant_disease_model.weights.h5"))

final_model_path = os.path.join(MODEL_DIR, "plant_disease_model.keras")
best_model_path = os.path.join(MODEL_DIR, "best_plant_disease_model.keras")
model.save(final_model_path)
model.save(best_model_path)

history = {
    key: [value for phase in histories for value in phase.get(key, [])]
    for key in {key for phase in histories for key in phase}
}
with open(os.path.join(MODEL_DIR, "training_history.json"), "w") as file:
    json.dump(history, file, indent=2)

epochs = range(1, len(history["accuracy"]) + 1)
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(epochs, history["accuracy"], label="train")
plt.plot(epochs, history["val_accuracy"], label="validation")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Training and validation accuracy")
plt.subplot(1, 2, 2)
plt.plot(epochs, history["loss"], label="train")
plt.plot(epochs, history["val_loss"], label="validation")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Training and validation loss")
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, "training_curves.png"), dpi=150)
plt.close()

loss, accuracy = model.evaluate(valid_ds, verbose=0)
print(f"\nFinal Validation Accuracy: {accuracy:.4f}")
print(f"Final Validation Loss: {loss:.4f}")