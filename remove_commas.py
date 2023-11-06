# Import the os, sys and pandas modules
import os
import sys
import pandas as pd

# Get the folder path from the command line argument
folder = sys.argv[1]
# Normalize and convert the folder path to an absolute path
folder = os.path.abspath(os.path.normpath(folder))
# Initialize an empty dictionary to store the old and new names
name_dict = {}
# Loop through the subdirectories, directories and files in the folder
for root, dirs, files in os.walk(folder):
    # Loop through the files
    for filename in files:
        # Check if the filename contains a comma
        if "," in filename:
            # Replace the comma with an empty string
            new_name = filename.replace(",", "")
            # Get the full path of the file
            file_path = os.path.join(root, filename)
            # Rename the file in the folder
            os.rename(file_path, os.path.join(root, new_name))
            # Add the old and new names to the dictionary
            name_dict[filename] = new_name
# Loop through the subdirectories, directories and files in the folder again
for root, dirs, files in os.walk(folder):
    # Loop through the files
    for filename in files:
        # Check if the file is a csv file
        if filename.endswith(".csv"):
            # Get the full path of the file
            file_path = os.path.join(root, filename)
            # Read the csv file as a pandas dataframe
            df = pd.read_csv(file_path)
            # Loop through the columns of the dataframe
            for col in df.columns:
                # Check if the column contains quoted paths
                if df[col].str.startswith('"').any() and df[col].str.endswith('"').any():
                    # Remove the quotes
                    df[col] = df[col].str[1:-1]
                    # Split the paths into directory and filename
                    df[col] = df[col].str.rsplit(os.sep, 1, expand=True)
                    # Replace the filenames with the new names if they are in the dictionary
                    df[col] = df[col].apply(lambda x: name_dict.get(x, x))
                    # Join the directory and filename
                    df[col] = df[col].str.join(os.sep)
                    # Add the quotes
                    df[col] = '"' + df[col] + '"'
            # Write the dataframe to the csv file
            df.to_csv(file_path, index=False)
