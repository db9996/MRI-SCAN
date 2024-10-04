import os
import pandas as pd

# Define the data directory where the image folders are located
data_dir = '/Users/jagathguru/Desktop/5C/data'  # Update this path if necessary
records = []

# Loop through each case directory
for case_id in os.listdir(data_dir):
    case_dir = os.path.join(data_dir, case_id)

    if os.path.isdir(case_dir):  # Ensure it is a directory
        for file in os.listdir(case_dir):
            if file.endswith('.tif') and '_mask' not in file:  # Image files
                image_path = os.path.join(case_id, file)
                mask_path = os.path.join(case_id, file.replace('.tif', '_mask.tif'))

                records.append({
                    'patient_id': case_id,
                    'image_path': image_path,
                    'mask_path': mask_path,
                    'mask': 0  # Adjust this if necessary
                })

# Create DataFrame and save to CSV
df = pd.DataFrame(records)
df.to_csv(os.path.join(data_dir, 'data_mask.csv'), index=False)

print(f"data_mask.csv created with {len(records)} records.")