import os
import pandas as pd
import numpy as np
from pathlib import Path

def create_comstock_structure(resolution):
    """
    Create the directory structure for ComStock data similar to CER format
    """
    base_path = Path(f"/home/vanshdhar/Desktop/ISP/TransApp/data/Comstock_{resolution}")
    
    # Create main directories
    inputs_dir = base_path / "Inputs"
    labels_dir = base_path / "Labels"
    
    inputs_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Created directory structure at: {base_path}")
    return base_path, inputs_dir, labels_dir

def process_comstock_labels(labels_file_path, labels_dir, resolution):
    """
    Process the ComStock labels CSV file and create separate files for each appliance
    
    Args:
        labels_file_path: Path to the comstock_15min_labels.csv file
        labels_dir: Directory to save the individual label files
        resolution: Resolution type (e.g., '15min')
    """
    # Read the labels file
    print(f"Reading ComStock {resolution} labels file...")
    labels_df = pd.read_csv(labels_file_path)
    
    # Display basic info about the dataset
    print(f"Labels shape: {labels_df.shape}")
    print(f"Columns: {labels_df.columns.tolist()}")
    print(f"Number of unique house IDs: {labels_df['id'].nunique()}")
    
    # Get appliance columns (all columns except 'id')
    appliance_columns = [col for col in labels_df.columns if col != 'id']
    
    print(f"Found {len(appliance_columns)} appliances: {appliance_columns}")
    
    # Create separate CSV files for each appliance
    for appliance in appliance_columns:
        # Create a dataframe with id and the specific appliance column
        appliance_df = labels_df[['id', appliance]].copy()
        
        # Rename columns to match CER format (id_pdl for house ID, appliance name for label)
        appliance_df = appliance_df.rename(columns={'id': 'id_pdl', appliance: appliance.replace('_ON', '_case')})
        
        # Set id_pdl as index
        appliance_df = appliance_df.set_index('id_pdl')
        
        # Save to CSV file
        output_file = labels_dir / f"{appliance.replace('_ON', '_case')}.csv"
        appliance_df.to_csv(output_file)
        
        # Display statistics for this appliance
        label_counts = appliance_df.iloc[:, 0].value_counts()
        print(f"  {appliance}: {dict(label_counts)} - Saved to {output_file.name}")
    
    return appliance_columns, labels_df['id'].unique().tolist()

def process_comstock_input_data(input_file_path, inputs_dir, house_ids, resolution):
    """
    Process the ComStock merged data file to create CER-format input data
    
    Args:
        input_file_path: Path to the comstock_merged_15min.csv file
        inputs_dir: Directory to save input files
        house_ids: List of house IDs from labels
        resolution: Resolution type (e.g., '15min')
    """
    print(f"Reading ComStock {resolution} input data file...")
    print(f"Input file path: {input_file_path}")
    if not os.path.exists(input_file_path):
        print(f"Error: Input data file not found at {input_file_path}")
        return None
    
    # Read the merged ComStock data file
    comstock_df = pd.read_csv(input_file_path)
    
    print(f"Comstock data shape: {comstock_df.shape}")
    print(f"Comstock columns: {list(comstock_df.columns[:10])}...")  # Show first 10 columns
    
    # Transform from ComStock format to CER format
    # ComStock format: Timestamp column + building columns
    # CER format: id_pdl column + time point columns (buildings as rows)
    
    # Drop the Timestamp column and transpose so buildings become rows
    if 'Timestamp' in comstock_df.columns:
        time_series_data = comstock_df.drop('Timestamp', axis=1).T
    else:
        # Assume first column is timestamp/index
        time_series_data = comstock_df.iloc[:, 1:].T
    
    # Reset index to get building IDs as a column
    time_series_data = time_series_data.reset_index()
    time_series_data.columns = ['id_pdl'] + list(range(len(comstock_df)))
    
    # Clean the building IDs to remove -0 suffix and convert to integers
    time_series_data['id_pdl'] = time_series_data['id_pdl'].str.replace('-0', '').astype(str)
    
    # Filter to only include buildings that have labels if house_ids provided
    if house_ids:
        house_ids_str = [str(hid) for hid in house_ids]
        time_series_data = time_series_data[time_series_data['id_pdl'].isin(house_ids_str)]
        print(f"Filtered to {len(time_series_data)} buildings with labels")
    
    # Set id_pdl as index to match CER format
    time_series_data = time_series_data.set_index('id_pdl')
    
    print(f"Final transformed data shape: {time_series_data.shape}")
    print(f"Number of buildings: {time_series_data.shape[0]}")
    print(f"Number of time points: {time_series_data.shape[1]}")
    print(f"Data range: {time_series_data.min().min():.3f} to {time_series_data.max().max():.3f}")
    
    # Save to CSV file with CER-like naming convention
    output_file = inputs_dir / f"x_comstock_{len(time_series_data)}_{resolution}.csv"
    time_series_data.to_csv(output_file)
    
    print(f"Created input file: {output_file.name}")
    print("Data format now matches CER structure:")
    print(f"  - Buildings as rows (index: id_pdl)")
    print(f"  - Time points as columns (0 to {time_series_data.shape[1]-1})")
    print(f"  - Shape: {time_series_data.shape}")
    
    return output_file, time_series_data

def create_metadata_file(base_path, appliances, house_ids, resolution, input_shape=None):
    """
    Create a metadata file with information about the dataset
    """
    metadata = {
        'dataset': 'ComStock',
        'resolution': resolution,
        'num_houses': len(house_ids),
        'num_appliances': len(appliances),
        'appliances': appliances,
        'house_id_range': f"{min(house_ids)} - {max(house_ids)}",
        'data_format': 'Similar to CER dataset format'
    }
    
    if input_shape is not None:
        metadata['time_points'] = input_shape[1]
        metadata['input_data_shape'] = f"{input_shape[0]} x {input_shape[1]}"
    
    metadata_file = base_path / "metadata.txt"
    with open(metadata_file, 'w') as f:
        for key, value in metadata.items():
            f.write(f"{key}: {value}\n")
    
    print(f"Created metadata file: {metadata_file}")

def main():
    """
    Main function to preprocess ComStock dataset for different resolutions
    """
    print("=== ComStock Dataset Preprocessing ===")
    
    # Define available resolutions and their corresponding files
    resolutions = {
        '15min': {
            'labels_file': "/home/vanshdhar/Desktop/ISP/TransApp/data/Comstock/comstock_15min_labels.csv",
            'input_file': "/home/vanshdhar/Desktop/ISP/TransApp/data/Comstock/comstock_merged_15_Min.csv"
        }
        # Add more resolutions here as needed
        # '1hour': {
        #     'labels_file': "/path/to/comstock_1hour_labels.csv",
        #     'input_file': "/path/to/comstock_merged_1hour.csv"
        # }
    }
    
    for resolution, files in resolutions.items():
        print(f"\n{'='*50}")
        print(f"Processing {resolution} resolution data...")
        print(f"{'='*50}")
        
        labels_file = files['labels_file']
        input_file = files['input_file']

        print(f"INput file: {input_file}")
        
        # Check if files exist
        if not os.path.exists(labels_file):
            print(f"Warning: Labels file not found at {labels_file}")
            print(f"Skipping {resolution} resolution...")
            continue
            
        if not os.path.exists(input_file):
            print(f"Warning: Input file not found at {input_file}")
            print(f"Skipping {resolution} resolution...")
            continue
        
        # Create directory structure for this resolution
        base_path, inputs_dir, labels_dir = create_comstock_structure(resolution)
        
        try:
            # Process labels
            appliances, house_ids = process_comstock_labels(labels_file, labels_dir, resolution)
            
            # Process input data with actual ComStock data
            input_file_created, input_df = process_comstock_input_data(
                input_file, inputs_dir, house_ids, resolution
            )
            
            if input_file_created and input_df is not None:
                # Create metadata
                create_metadata_file(base_path, appliances, house_ids, resolution, input_df.shape)
                
                print(f"\n=== {resolution.upper()} Preprocessing Complete ===")
                print(f"Created {len(appliances)} appliance label files")
                print(f"Processed {len(house_ids)} house IDs")
                print(f"Input data shape: {input_df.shape}")
                print(f"Files saved in: {base_path}")
            else:
                print(f"Failed to process input data for {resolution}")
                
        except Exception as e:
            print(f"Error during {resolution} processing: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*50}")
    print("All resolutions processed!")
    print("Next steps:")
    print("1. Verify the generated input files contain correct time series data")
    print("2. Test data loading with the TransApp framework")
    print("3. Update data_utils.py to support ComStock data loading")

if __name__ == "__main__":
    main()
