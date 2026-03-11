import pandas as pd
import numpy as np
from pathlib import Path
import os
import argparse

# Configuration
BASE_DIR = Path("/workspace/datdq/SignWeather")
METADATA_DIR = BASE_DIR / "data" / "metadata"
OUTPUT_DIR = BASE_DIR / "data" / "lists"

# Default metadata filename (can be overridden with --metadata)
DEFAULT_METADATA_FILENAME = "scene_metadata.csv"


def create_train_val_test_split(csv_path, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_seed=42):
    """
    Chia dataset thành train/val/test với tỷ lệ tùy chỉnh.
    Đảm bảo phân bố đều theo quality_level và content_label.
    """
    print(f"Loading dataset from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Total samples: {len(df)}")

    # Validate ratios
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Train + Val + Test ratios must sum to 1.0")

    # Set random seed for reproducibility
    np.random.seed(random_seed)

    # Stratified split based on quality_level and content_label
    stratify_cols = ['quality_level', 'content_label']
    available_cols = [col for col in stratify_cols if col in df.columns]

    if available_cols:
        print(f"Performing stratified split based on: {available_cols}")
        # Create stratification groups
        df['stratify_group'] = df[available_cols].astype(str).agg('_'.join, axis=1)

        # Get unique groups and their sizes
        groups = df['stratify_group'].value_counts()
        print(f"Stratification groups: {len(groups)}")

        train_indices = []
        val_indices = []
        test_indices = []

        for group, group_df in df.groupby('stratify_group'):
            group_size = len(group_df)
            indices = group_df.index.tolist()
            np.random.shuffle(indices)

            # Calculate split sizes for this group
            n_train = max(1, int(group_size * train_ratio))
            n_val = max(1, int(group_size * val_ratio))
            n_test = group_size - n_train - n_val

            # Adjust if necessary to ensure at least 1 sample per split
            if n_test < 1 and n_val > 1:
                n_val -= 1
                n_test = 1
            elif n_test < 1 and n_val == 1:
                n_train -= 1
                n_test = 1

            train_indices.extend(indices[:n_train])
            val_indices.extend(indices[n_train:n_train + n_val])
            test_indices.extend(indices[n_train + n_val:])

        # Create split DataFrames
        train_df = df.loc[train_indices].copy()
        val_df = df.loc[val_indices].copy()
        test_df = df.loc[test_indices].copy()

    else:
        print("No stratification columns found, performing random split")
        # Random split
        indices = df.index.tolist()
        np.random.shuffle(indices)

        n_total = len(indices)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)

        train_indices = indices[:n_train]
        val_indices = indices[n_train:n_train + n_val]
        test_indices = indices[n_train + n_val:]

        train_df = df.loc[train_indices].copy()
        val_df = df.loc[val_indices].copy()
        test_df = df.loc[test_indices].copy()

    # Add split column to DataFrames
    train_df['split'] = 'train'
    val_df['split'] = 'val'
    test_df['split'] = 'test'

    # Combine all splits into one DataFrame
    combined_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

    # Remove stratify_group column if it exists
    if 'stratify_group' in combined_df.columns:
        combined_df = combined_df.drop('stratify_group', axis=1)

    # Print statistics
    print("\n" + "="*60)
    print("SPLIT STATISTICS:")
    print("="*60)
    print(f"Train set: {len(train_df)} samples ({len(train_df)/len(df)*100:.1f}%)")
    print(f"Val set:   {len(val_df)} samples ({len(val_df)/len(df)*100:.1f}%)")
    print(f"Test set:  {len(test_df)} samples ({len(test_df)/len(df)*100:.1f}%)")

    # Split distribution
    print("\nSplit Distribution in Combined CSV:")
    print(combined_df['split'].value_counts().to_dict())

    # Quality distribution
    if 'quality_level' in df.columns:
        print("\nQuality Level Distribution:")
        print("Train:", train_df['quality_level'].value_counts().to_dict())
        print("Val:  ", val_df['quality_level'].value_counts().to_dict())
        print("Test: ", test_df['quality_level'].value_counts().to_dict())

    if 'content_label' in df.columns:
        print("\nContent Label Distribution:")
        print("Train:", train_df['content_label'].value_counts().to_dict())
        print("Val:  ", val_df['content_label'].value_counts().to_dict())
        print("Test: ", test_df['content_label'].value_counts().to_dict())

    # Save splits
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Save individual split CSVs
    train_df.to_csv(OUTPUT_DIR / "train.csv", index=False, encoding='utf-8-sig')
    val_df.to_csv(OUTPUT_DIR / "val.csv", index=False, encoding='utf-8-sig')
    test_df.to_csv(OUTPUT_DIR / "test.csv", index=False, encoding='utf-8-sig')

    # Save combined CSV with split column
    combined_csv_path = METADATA_DIR / "vswd_final_split.csv"
    combined_df.to_csv(combined_csv_path, index=False, encoding='utf-8-sig')

    # Save path lists for easy loading
    with open(OUTPUT_DIR / "train_paths.txt", 'w', encoding='utf-8') as f:
        for path in train_df['path']:
            f.write(f"{path}\n")

    with open(OUTPUT_DIR / "val_paths.txt", 'w', encoding='utf-8') as f:
        for path in val_df['path']:
            f.write(f"{path}\n")

    with open(OUTPUT_DIR / "test_paths.txt", 'w', encoding='utf-8') as f:
        for path in test_df['path']:
            f.write(f"{path}\n")

    print(f"\nFiles saved to {OUTPUT_DIR}:")
    print("- train.csv, val.csv, test.csv (individual splits)")
    print("- train_paths.txt, val_paths.txt, test_paths.txt (path lists)")
    print(f"- {combined_csv_path} (original CSV with 'split' column added)")

    return train_df, val_df, test_df, combined_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create train/val/test split for VSWD dataset")
    parser.add_argument("--metadata", type=str, default=DEFAULT_METADATA_FILENAME, help="Metadata CSV filename under data/metadata (default: scene_metadata.csv)")
    parser.add_argument("--output", type=str, default="vswd_final_split.csv", help="Combined output CSV filename under data/metadata (default: vswd_final_split.csv)")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train ratio (default: 0.8)")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Val ratio (default: 0.1)")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="Test ratio (default: 0.1)")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed (default: 42)")

    args = parser.parse_args()

    # Resolve metadata path and output path
    CSV_PATH = METADATA_DIR / args.metadata
    OUTPUT_CSV = METADATA_DIR / args.output

    print("🔀 VSWD Dataset Train/Val/Test Split")
    print(f"📁 Metadata: {CSV_PATH}")
    print(f"📤 Combined output: {OUTPUT_CSV}")
    print(f"📊 Ratios: Train={args.train_ratio}, Val={args.val_ratio}, Test={args.test_ratio}")

    if not CSV_PATH.exists():
        print(f"❌ Error: Metadata CSV not found at {CSV_PATH}")
        exit(1)

    try:
        train_df, val_df, test_df, combined_df = create_train_val_test_split(
            str(CSV_PATH),
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            random_seed=args.random_seed
        )

        # Save combined to user-specified output filename
        combined_df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
        print("\n✅ Split completed successfully!")
    except Exception as e:
        print(f"❌ Error: {e}")
        exit(1)
