import csv
import json
import os
import re
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from predict import IMG_SIZE, MODEL_PATH, CLASS_NAMES_PATH, preprocess_leaf_image


BASE_DIR = Path(__file__).resolve().parent
TEST_DIR = BASE_DIR / "test"
DEFAULT_DATA_DIR = BASE_DIR / "Segmented_Dataset"
if not DEFAULT_DATA_DIR.exists() and Path(r"C:\cropai-data\NewPlantDiseases").exists():
    DEFAULT_DATA_DIR = Path(r"C:\cropai-data\NewPlantDiseases")
DATA_DIR = Path(os.environ.get("CROP_DATA_DIR", str(DEFAULT_DATA_DIR)))
OUTPUT_DIR = BASE_DIR / "evaluation_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def normalize(text):
    return re.sub(r"[^a-z0-9]", "", text.lower())


def actual_class(filename):
    name = normalize(Path(filename).stem)
    mappings = {
        "applecedarrust": "Apple___Cedar_apple_rust",
        "applescab": "Apple___Apple_scab",
        "corncommonrust": "Corn_(maize)___Common_rust_",
        "potatoearlyblight": "Potato___Early_blight",
        "potatohealthy": "Potato___healthy",
        "tomatoearlyblight": "Tomato___Early_blight",
        "tomatohealthy": "Tomato___healthy",
        "tomatoyellowcurlvirus": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    }
    for prefix, class_name in mappings.items():
        if name.startswith(prefix):
            return class_name
    return None


def count_classes(root):
    counts = {}
    root = Path(root)
    if not root.exists():
        return counts
    for class_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        counts[class_dir.name] = sum(
            file.suffix.lower() in SUPPORTED_EXTENSIONS
            for file in class_dir.rglob("*")
            if file.is_file()
        )
    return counts


def print_mapping(class_names):
    print(f"Number of classes: {len(class_names)}")
    print("Class mapping:")
    for index, class_name in enumerate(class_names):
        print(f"{index} -> {class_name}")


def main():
    print(f"Loading model: {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    with open(CLASS_NAMES_PATH, "r") as file:
        class_names = json.load(file)

    if model.output_shape[-1] != len(class_names):
        raise ValueError("The model output and class_names.json mapping do not match.")

    print_mapping(class_names)
    train_counts = count_classes(DATA_DIR / "train")
    valid_counts = count_classes(DATA_DIR / "valid")
    test_files = sorted(
        file for file in TEST_DIR.iterdir()
        if file.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    print(f"Training images: {sum(train_counts.values())}")
    print(f"Validation images: {sum(valid_counts.values())}")
    print(f"Test images: {len(test_files)}")
    print("Training counts:")
    for class_name in class_names:
        print(f"  {class_name}: {train_counts.get(class_name, 0)}")
    print("Validation counts:")
    for class_name in class_names:
        print(f"  {class_name}: {valid_counts.get(class_name, 0)}")

    y_true = []
    y_pred = []
    confidences = []
    rows = []

    for image_path in test_files:
        expected = actual_class(image_path.name)
        if expected is None:
            print(f"SKIPPED: no label mapping for {image_path.name}")
            continue
        with Image.open(image_path) as image:
            probabilities = model.predict(preprocess_leaf_image(image), verbose=0)[0]
        index = int(np.argmax(probabilities))
        predicted = class_names[index]
        confidence = float(probabilities[index])
        y_true.append(expected)
        y_pred.append(predicted)
        confidences.append(confidence)
        rows.append((image_path.name, expected, predicted, confidence))
        print(
            f"{'OK' if expected == predicted else 'MISS'} {image_path.name}: "
            f"actual={expected}; predicted={predicted}; confidence={confidence * 100:.2f}%"
        )

    if not y_true:
        raise RuntimeError("No labeled test images were found.")

    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=class_names, zero_division=0
    )
    report = classification_report(
        y_true, y_pred, labels=class_names, zero_division=0, digits=4
    )
    matrix = confusion_matrix(y_true, y_pred, labels=class_names)

    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Macro precision: {precision.mean() * 100:.2f}%")
    print(f"Macro recall: {recall.mean() * 100:.2f}%")
    print(f"Macro F1: {f1.mean() * 100:.2f}%")
    print(f"Mean confidence: {np.mean(confidences) * 100:.2f}%")
    print("\nClassification report:\n" + report)
    print("Per-class accuracy (recall):")
    for name, value, count in zip(class_names, recall, support):
        print(f"  {name}: {value * 100:.2f}% ({int(count)} images)")

    with open(OUTPUT_DIR / "predictions.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["filename", "actual", "predicted", "confidence"])
        writer.writerows(rows)

    np.savetxt(OUTPUT_DIR / "confusion_matrix.csv", matrix, fmt="%d", delimiter=",")
    with open(OUTPUT_DIR / "classification_report.txt", "w") as file:
        file.write(report)

    print(f"Reports written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
