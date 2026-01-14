#!/usr/bin/env python3
"""
SummEval Dataset Cleaning Tool.

Converts the original SummEval dataset from one-to-many format 
(one document with multiple summaries) to one-to-one format
(each document-summary pair as a separate sample).

Designed for Google Colab environment with Google Drive integration.
"""

import os
import json
import argparse
import pandas as pd
import numpy as np
from datasets import Dataset
from tqdm import tqdm

# Environment detection
try:
    from google.colab import drive
    IN_COLAB = True
except ImportError:
    IN_COLAB = False


def mount_google_drive():
    """
    Mount Google Drive in Colab environment.
    
    Returns:
        True if mounted successfully, False otherwise.
    """
    if not IN_COLAB:
        print("Not running in Colab environment")
        return False
    
    try:
        print("Mounting Google Drive...")
        drive.mount('/content/drive')
        print("Google Drive mounted successfully")
        return True
    except Exception as e:
        print(f"Failed to mount Google Drive: {e}")
        return False


def load_summeval_from_drive(drive_dir="/content/drive/MyDrive/ML_datasets/summeval"):
    """
    Load summeval_test.parquet file from Google Drive.
    
    Args:
        drive_dir: Dataset directory in Google Drive.
    
    Returns:
        Path to parquet file, or None if not found.
    """
    if not IN_COLAB:
        print("Cannot load from Google Drive outside Colab")
        return None
    
    parquet_path = os.path.join(drive_dir, "summeval_test.parquet")
    
    print(f"Looking for: {parquet_path}")
    
    if os.path.exists(parquet_path):
        print(f"Found: {parquet_path}")
        return parquet_path
    else:
        print(f"File not found: {parquet_path}")
        return None


def clean_summeval_dataset(
    data_path,
    output_format="parquet",
    output_dir="./data",
    drive_output_dir=None
):
    """
    Clean SummEval dataset to one-to-one format.
    
    Converts the original format where each document has multiple summaries
    into a flat format where each row is a document-summary pair.
    
    Args:
        data_path: Path to input parquet file.
        output_format: Output format (parquet, csv, json, dataset).
        output_dir: Local output directory.
        drive_output_dir: Google Drive output directory.
    
    Returns:
        Cleaned dataset as HuggingFace Dataset object.
    """
    print(f"Cleaning SummEval dataset: {data_path}")
    
    # Read input file
    print("Reading parquet file...")
    df = pd.read_parquet(data_path)
    print(f"Loaded {len(df)} documents")
    
    # Initialize output structure
    cleaned_data = {
        "text": [],
        "summary": [],
        "relevance": [],
        "coherence": [],
        "fluency": [],
        "consistency": []
    }
    
    total_samples = 0
    data = df.to_dict('records')
    
    for item in tqdm(data, desc="Processing documents"):
        doc_text = item.get("text", "")
        
        machine_summaries = item.get("machine_summaries", [])
        relevance_scores = item.get("relevance", [])
        coherence_scores = item.get("coherence", [])
        fluency_scores = item.get("fluency", [])
        consistency_scores = item.get("consistency", [])
        
        num_summaries = len(machine_summaries)
        
        # Validate score counts
        if (len(relevance_scores) != num_summaries or
            len(coherence_scores) != num_summaries or
            len(fluency_scores) != num_summaries or
            len(consistency_scores) != num_summaries):
            print(f"Warning: Score count mismatch, skipping document")
            continue
        
        # Create one row per summary
        for i in range(num_summaries):
            cleaned_data["text"].append(doc_text)
            cleaned_data["summary"].append(machine_summaries[i])
            cleaned_data["relevance"].append(float(relevance_scores[i]))
            cleaned_data["coherence"].append(float(coherence_scores[i]))
            cleaned_data["fluency"].append(float(fluency_scores[i]))
            cleaned_data["consistency"].append(float(consistency_scores[i]))
            total_samples += 1
    
    print(f"Generated {total_samples} document-summary pairs")
    
    # Create DataFrame
    cleaned_df = pd.DataFrame(cleaned_data)
    
    # Print statistics
    print("\nDataset statistics:")
    print(f"  Total samples: {len(cleaned_df)}")
    print(f"  Relevance range: {cleaned_df['relevance'].min():.2f} - {cleaned_df['relevance'].max():.2f}")
    print(f"  Coherence range: {cleaned_df['coherence'].min():.2f} - {cleaned_df['coherence'].max():.2f}")
    print(f"  Fluency range: {cleaned_df['fluency'].min():.2f} - {cleaned_df['fluency'].max():.2f}")
    print(f"  Consistency range: {cleaned_df['consistency'].min():.2f} - {cleaned_df['consistency'].max():.2f}")
    
    # Save to Google Drive
    if drive_output_dir and IN_COLAB:
        print(f"Saving to Google Drive: {drive_output_dir}")
        os.makedirs(drive_output_dir, exist_ok=True)
        
        if output_format == "parquet":
            output_path = os.path.join(drive_output_dir, "summeval_cleaned.parquet")
            cleaned_df.to_parquet(output_path, index=False)
        elif output_format == "csv":
            output_path = os.path.join(drive_output_dir, "summeval_cleaned.csv")
            cleaned_df.to_csv(output_path, index=False)
        elif output_format == "json":
            output_path = os.path.join(drive_output_dir, "summeval_cleaned.json")
            cleaned_df.to_json(output_path, orient="records", lines=True)
        
        print(f"Saved: {output_path}")
    
    # Create Dataset object
    dataset = Dataset.from_pandas(cleaned_df)
    
    if output_format == "dataset" and drive_output_dir:
        dataset_path = os.path.join(drive_output_dir, "summeval_cleaned")
        dataset.save_to_disk(dataset_path)
        print(f"Saved as Dataset: {dataset_path}")
    
    return dataset


def view_data(data_path):
    """
    Display basic information about the dataset.
    
    Args:
        data_path: Path to parquet file.
    """
    print("\n===== Dataset Info =====")
    try:
        df = pd.read_parquet(data_path)
        print(f"Rows: {len(df)}")
        print(f"Columns: {df.columns.tolist()}")
        
        if len(df) > 0:
            print("\nFirst sample:")
            first = df.iloc[0].to_dict()
            for key, value in first.items():
                if isinstance(value, list):
                    print(f"  {key}: {value[:3]}... ({len(value)} items)")
                else:
                    print(f"  {key}: {value}")
    except Exception as e:
        print(f"Error reading data: {e}")


def main():
    parser = argparse.ArgumentParser(description="SummEval Dataset Cleaner")
    parser.add_argument(
        "--drive_dir",
        type=str,
        default="/content/drive/MyDrive/ML_datasets/summeval",
        help="Google Drive dataset directory"
    )
    parser.add_argument(
        "--output_format",
        type=str,
        default="parquet",
        choices=["parquet", "csv", "json", "dataset"],
        help="Output format"
    )
    
    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"Ignoring unknown arguments: {unknown}")
    
    if not IN_COLAB:
        print("Error: This script requires Google Colab environment")
        return
    
    if not mount_google_drive():
        print("Error: Failed to mount Google Drive")
        return
    
    data_path = load_summeval_from_drive(args.drive_dir)
    
    if data_path is None or not os.path.exists(data_path):
        print("Error: summeval_test.parquet not found")
        return
    
    view_data(data_path)
    
    dataset = clean_summeval_dataset(
        data_path,
        args.output_format, 
        output_dir="./data",
        drive_output_dir=args.drive_dir
    )
    
    # Display cleaned samples
    print("\n===== Cleaned Samples =====")
    for i in range(min(5, len(dataset))):
        sample = dataset[i]
        print(f"\nSample {i+1}:")
        print(f"  Document: {sample['text'][:100]}...")
        print(f"  Summary: {sample['summary']}")
        print(f"  Scores: rel={sample['relevance']:.2f}, coh={sample['coherence']:.2f}, "
              f"flu={sample['fluency']:.2f}, con={sample['consistency']:.2f}")
        print("-" * 60)


if __name__ == "__main__":
    main() 
