# Sanitize a folder of generated images and prompts for consistency.
#
# This code is to go through the output of the image generators and the prompt files.
# To prepare for human evaluation.
# There are CSV files that list the filenames.
# It cleans up filenames that have a comma in them, removing the commas, and ensuring there aren't double underscores.
# The filenames also contain the generator seed, which ensures nearly identical images are generated, this code reads those numbers and adds them to the CSV.
# Since this code modifies files directly on disk, it asks for user confirmation before making the changes.

import os
import sys
import json
import pandas as pd
import re
import argparse
import copy
import csv


def get_seed(filename):
    # Match the last group of digits before the .png extension
    match = re.search(r"(\d+)\.png$", filename)
    if match:
        return int(match.group(1))
    return None


parser = argparse.ArgumentParser()
parser.add_argument("folder", default="CCUB_eval", help="The folder path")
parser.add_argument("--auto", action="store_true", help="Automatically apply changes to the filesystem")
args = parser.parse_args()


folder = os.path.abspath(os.path.normpath(args.folder))
folder_name = os.path.basename(folder)
parent_dir = os.path.dirname(folder)
json_file = f"rename_{folder_name}.json"
json_path = os.path.join(parent_dir, json_file)
name_dict = {}
csv_list = []
png_count = 0
not_in_dict = 0
files_in_csv_but_not_on_disk = 0
png_files_missing_seed = 0
png_set = set()
csv_set = set()
if os.path.exists(json_path):
    with open(json_path) as f:
        name_dict = json.load(f)
for root, dirs, files in os.walk(folder):
    for filename in files:
        file_path = os.path.join(root, filename)
        if os.path.isfile(file_path):
            if filename.endswith(".csv"):
                csv_list.append(file_path)
            elif filename.endswith(".png"):
                png_set.add(filename)
                if "," in filename:
                    new_name = filename.replace(",_", "_").replace(',','_').replace("__", "_")
                    name_dict[filename] = new_name
                    with open(json_path, "w") as f:
                        json.dump(name_dict, f, indent=4)
                # else:
                #     name_dict[filename] = filename
df_list = []
step = 0
for csv_path in csv_list:
    # Use quotechar to handle the double quotes in the CSV files
    df = pd.read_csv(csv_path, quotechar='"')
    # Check if there is an existing seed column
    if "seed" not in df.columns:
        df["seed"] = ""
    for index, row in df.iterrows():
        seeds = []
        for col in row.index:
            step += 1
            # Check if the column contains a PNG file
            if ".png" in row[col]:
                dir, filename = os.path.split(row[col])
                # Use os.path.splitext to get the file extension
                name, ext = os.path.splitext(filename)
                if ext == ".png":
                    csv_set.add(filename)
                    png_count += 1
                    # Check if the filename is valid and exists in the folder
                    file_to_check = os.path.join(folder, row[col])
                    # print(f'is a png: {file_to_check} <<<<<<')
                    if os.path.isfile(file_to_check):
                        # print('filename exists on disk >>>>>>')
                        if filename in name_dict:
                            # print(f'filename exists in dict: {filename}:{name_dict[filename]} >>>>>>')
                            # print(f'{step} >>> row[col] before: {row[col]}')
                            df.loc[index, col] = row[col].replace(filename, name_dict[filename])
                            # print(f'{step} >>> row[col] after: {row[col]}')
                        else:
                            not_in_dict += 1
                        # Use filename instead of row[col] to get the seed
                        seed = get_seed(filename)
                        print("filename: {filename} seed: {seed}")
                        # Check if the seed is not None before appending it
                        if seed is not None:
                            seeds.append(seed)
                        else:
                            print("Warning: The file {filename} was missing a generator seed.")
                            png_files_missing_seed += 1
                    else:
                        # Print an error message
                        print(f"Error: {filename} is not a valid or existing file.")
                        files_in_csv_but_not_on_disk += 1
                    # row[col] = os.path.join(parent_dir, row[col])
        # Use df.drop_duplicates to remove duplicate rows
        unique_seeds = pd.Series(seeds).drop_duplicates()
        if len(unique_seeds) == 1:
            seed = unique_seeds[0]
            # Convert the seed to a string before assigning it
            df.loc[index, "seed"] = str(seed)
        else:
            # Print a warning with the csv file name, the row entry and the seeds
            print(f"Warning: The seeds in row {index} of {csv_path} are not consistent. The row entry is: {row}. The seeds are: {seeds}.")
    df_list.append(copy.deepcopy(df))
percentage = round((not_in_dict / png_count) * 100, 2) if png_count > 0 else 0
print("The following files will be renamed:")
for old_name, new_name in name_dict.items():
    print(f"{old_name} -> {new_name}")
set_diff = png_set - csv_set
print("")
print("Results of Dataset Cleanup and Summary Stats")
print("--------------------------------------------")
if len(set_diff) or files_in_csv_but_not_on_disk:
    print(f"The following PNG files are not in the CSV files: {set_diff}. This means that some of the PNG files in the folder are not referenced in the CSV files.")
    print(f"Error: The following is an error if greater than zero: {files_in_csv_but_not_on_disk} files are in a csv file but could not be found on the disk.")
else:
    print("Good: All PNG files in the CSV files were successfully found on disk.")

if png_files_missing_seed:
    print(f"Warning: {png_files_missing_seed} png files are missing a generator seed value.")
else:
    print("Good: All png files have a generator seed.")
print(f"Files with commas: {not_in_dict} PNG files, or {percentage}% of the total {png_count} PNG files are not in the dictionary of files to rename. This means that their PNG files in the folder did not need to be renamed.")
# Use the set difference to show the PNG files that are not in the CSV files
print(f"There are {len(csv_list)} CSV files and {len(name_dict)} PNG files to be modified.")

if args.auto or input("Do you want to rename the files in the folder and save the CSV files? (y/n) ").lower() == "y":
    for root, dirs, files in os.walk(folder):
        for filename in files:
            if filename in name_dict:
                file_path = os.path.join(root, filename)
                new_name = name_dict[filename]
                os.rename(file_path, os.path.join(root, new_name))
                print(f"Renamed {filename} to {new_name}.")
    for csv_path, df in zip(csv_list, df_list):
        df.to_csv(csv_path, index=False, quoting=csv.QUOTE_ALL)
        print(f"Saved {csv_path}.")
else:
    print("No files were renamed or saved.")
