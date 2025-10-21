#!/usr/bin/env python3

import pandas as pd
import os
from pathlib import Path

def append_csv_files():
    """
    Append the 3 CSV files in the data directory while preserving structure.
    Handles different column structures by standardizing them.
    """
    
    data_dir = Path("/Users/seangarcia/Downloads/reddit_scraper/data")
    files = [
        "993octoberposts.csv",
        "new_posts_9_30.csv", 
        "past3months.csv"
    ]
    
    standard_columns = [
        "Post ID", "Summary", "Full Title", "Full Content", 
        "Timestamp", "Score", "Comments", "URL", "Is Report"
    ]
    
    combined_data = []
    
    for file in files:
        file_path = data_dir / file
        print(f"Processing {file}...")
        
        try:
            df = pd.read_csv(file_path)
            print(f"  - Shape: {df.shape}")
            print(f"  - Columns: {list(df.columns)}")
            
            if "Title & Content" in df.columns:
                df_standardized = pd.DataFrame()
                df_standardized["Post ID"] = [f"legacy_{i}" for i in range(len(df))]
                df_standardized["Summary"] = df["Title & Content"].str[:100] + "..."  # Truncate for summary
                df_standardized["Full Title"] = df["Title & Content"]
                df_standardized["Full Content"] = df["Title & Content"]
                df_standardized["Timestamp"] = df["Timestamp"]
                df_standardized["Score"] = df["Score"]
                df_standardized["Comments"] = df["Comments"]
                df_standardized["URL"] = df["URL"]
                df_standardized["Is Report"] = df["Is Report"]
            else:
                df_standardized = df.copy()
            
            for col in standard_columns:
                if col not in df_standardized.columns:
                    df_standardized[col] = ""
            
            df_standardized = df_standardized[standard_columns]
            
            df_standardized["Source File"] = file
            
            combined_data.append(df_standardized)
            print(f"  - Processed {len(df_standardized)} rows")
            
        except Exception as e:
            print(f"  - Error processing {file}: {e}")
            continue
    
    if combined_data:
        final_df = pd.concat(combined_data, ignore_index=True)
        
        output_path = data_dir / "combined_posts.csv"
        final_df.to_csv(output_path, index=False)
        
        print(f"\nCombined file saved to: {output_path}")
        print(f"Total rows: {len(final_df)}")
        print(f"Columns: {list(final_df.columns)}")
        
        print("\nSummary by source file:")
        print(final_df["Source File"].value_counts())
        
        return output_path
    else:
        print("No data to combine!")
        return None

if __name__ == "__main__":
    append_csv_files()
