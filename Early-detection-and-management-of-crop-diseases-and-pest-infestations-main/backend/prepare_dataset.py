import argparse
from pathlib import Path

from datasets import load_dataset


DATASET_ID = "Hemg/new-plant-diseases-dataset"
SEED = 42


def write_split(dataset, split_name, output_dir):
    labels = dataset.features["label"].names
    split_dir = output_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    for index, item in enumerate(dataset):
        class_dir = split_dir / labels[item["label"]]
        class_dir.mkdir(parents=True, exist_ok=True)
        image = item["image"].convert("RGB")
        image.save(class_dir / f"{split_name}_{index:06d}.jpg", quality=95)
        if (index + 1) % 1000 == 0:
            print(f"{split_name}: {index + 1:,}/{len(dataset):,}")


def main():
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default=str(script_dir / "New Plant Diseases Dataset(Augmented)"),
        help="Directory that will contain train and valid class folders.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output).resolve()
    print(f"Loading {DATASET_ID}...")
    dataset = load_dataset(DATASET_ID, split="train")
    split = dataset.train_test_split(
        test_size=0.2,
        seed=SEED,
        stratify_by_column="label",
    )

    print(f"Writing dataset to {output_dir}")
    write_split(split["train"], "train", output_dir)
    write_split(split["test"], "valid", output_dir)
    print("Dataset preparation completed.")


if __name__ == "__main__":
    main()
