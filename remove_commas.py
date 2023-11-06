# Import the os, sys, json, pandas and re modules
import os
import sys
import json
import pandas as pd
import re

# Define a function to extract the random seed from the filename
def get_seed(filename):
    # Use a regular expression to find the number in the filename
    match = re.search(r"\d+", filename)
    # Check if the match is not None
    if match:
        # Return the matched number as an integer
        return int(match.group())
    # Return None
    return None

# Get the folder path from the command line argument or use the default value
folder = sys.argv[1] if len(sys.argv) > 1 else "CCUB_eval"
# Normalize and convert the folder path to an absolute path
folder = os.path.abspath(os.path.normpath(folder))
# Get the folder name
folder_name = os.path.basename(folder)
# Get the parent directory of the folder
parent_dir = os.path.dirname(folder)
# Get the JSON file name
json_file = "rename_" + folder_name + ".json"
# Get the JSON file path
json_path = os.path.join(parent_dir, json_file)
# Initialize an empty dictionary to store the old and new names
name_dict = {}
# Initialize an empty list to store the csv paths
csv_list = []
# Initialize a counter to store the number of PNG files
png_count = 0
# Initialize a counter to store the number of PNG files that are not in the dictionary
not_in_dict = 0
# Initialize a set to store the PNG filenames in the folder
png_set = set()
# Initialize a set to store the PNG filenames in the csv files
csv_set = set()
# Check if the JSON file exists
if os.path.exists(json_path):
    # Load the JSON file
    with open(json_path) as f:
        name_dict = json.load(f)
# Loop through the subdirectories, directories and files in the folder
for root, dirs, files in os.walk(folder):
    # Loop through the files
    for filename in files:
        # Get the full path of the file
        file_path = os.path.join(root, filename)
        # Check if the file path is valid
        if os.path.isfile(file_path):
            # Check if the file is a csv file
            if filename.endswith(".csv"):
                # Add the file path to the csv list
                csv_list.append(file_path)
            # Check if the file is a PNG file
            elif filename.endswith(".png"):
                # Add the filename to the PNG set
                png_set.add(filename)
            # Check if the filename contains a comma
            elif "," in filename:
                # Replace the comma with an empty string
                new_name = filename.replace(",", "")
                # Add the old and new names to the dictionary
                name_dict[filename] = new_name
# Save the dictionary to the JSON file
with open(json_path, "w") as f:
    json.dump(name_dict, f)
# Initialize an empty list to store the dataframes
df_list = []
# Loop through the csv paths in the csv list
for csv_path in csv_list:
    # Read the csv file as a pandas dataframe
    df = pd.read_csv(csv_path)
    # Create a column named "seed" and assign an empty value
    df["seed"] = ""
    # Loop through the rows of the dataframe
    for index, row in df.iterrows():
        # Initialize a list to store the seeds in the row
        seeds = []
        # Loop through the columns of the row
        for col in row.index:
            # Check if the value is a quoted path
            if isinstance(row[col], str) and row[col].startswith('"') and row[col].endswith('"'):
                # Remove the quotes
                row[col] = row[col][1:-1]
                # Split the path into directory and filename
                dir, filename = os.path.split(row[col])
                # Check if the filename is a PNG file
                if filename.endswith(".png"):
                    # Add the filename to the CSV set
                    csv_set.add(filename)
                    # Increment the PNG file counter
                    png_count += 1
                    # Check if the filename is in the dictionary
                    if filename in name_dict:
                        # Replace the filename with the new name
                        filename = name_dict[filename]
                        # Join the directory and filename
                        row[col] = os.path.join(dir, filename)
                    else:
                        # Increment the not in dictionary counter
                        not_in_dict += 1
                        # Print a warning
                        print(f"Warning: {filename} is not in the dictionary.")
                    # Extract the random seed from the filename
                    seed = get_seed(filename)
                    # Add the seed to the list
                    seeds.append(seed)
        # Check if the seeds in the row are consistent
        if len(set(seeds)) == 1:
            # Assign the seed value to the variable
            seed = seeds[0]
            # Assign the seed value to the column
            df.loc[index, "seed"] = seed
        else:
            # Print a warning
            print(f"Warning: The seeds in row {index} are not consistent.")
    # Add the dataframe to the list
    df_list.append(df)
# Calculate the percentage of PNG files that are not in the dictionary
percentage = round((not_in_dict / png_count) * 100, 2) if png_count > 0 else 0
# Print the percentage and the total number of PNG files
print(f"{not_in_dict} or {percentage}% of {png_count} PNG files are not in the dictionary. This means that some of the PNG files in the folder have not been renamed or do not exist.")
# Print the number of PNG files that are not in the CSV files
print(f"There are {len(png_set - csv_set)} PNG files that are not in the CSV files. This means that some of the PNG files in the folder are not referenced in the CSV files.")
# Print the number of CSV files and PNG files to be modified
print(f"There are {len(csv_list)} CSV files and {len(name_dict)} PNG files to be modified.")
# Print the files to be renamed
print("The following files will be renamed:")
for old_name, new_name in name_dict.items():
    print(f"{old_name} -> {new_name}")
# Ask for confirmation before renaming the files and saving the csv files
answer = input("Do you want to rename the files in the folder and save the csv files? (y/n) ")
if answer.lower() == "y":
    # Loop through the subdirectories, directories and files in the folder
    for root, dirs, files in os.walk(folder):
        # Loop through the files
        for filename in files:
            # Check if the filename is in the dictionary
            if filename in name_dict:
                # Get the full path of the file
                file_path = os.path.join(root, filename)
                # Get the new name from the dictionary
                new_name = name_dict[filename]
                # Rename the file in the folder
                os.rename(file_path, os.path.join(root, new_name))
                # Print a message
                print(f"Renamed {filename} to {new_name}.")
    # Loop through the csv paths and dataframes in the csv list and df list
    for csv_path, df in zip(csv_list, df_list):
        # Write the dataframe to the csv file
        df.to_csv(csv_path, index=False)
        # Print a message
        print(f"Saved {csv_path}.")
else:
    # Print a message
    print("No files were renamed or saved.")
