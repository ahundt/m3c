import os
import sys
import json
import pandas as pd
import re
import argparse

def get_seed(filename):
    match = re.search(r"\d+", filename)
    if match:
        return int(match.group())
    return None

parser = argparse.ArgumentParser()
parser.add_argument("folder", default="CCUB_eval", help="The folder path")
args = parser.parse_args()

folder = os.path.abspath(os.path.normpath(args.folder))
folder_name = os.path.basename(folder)
parent_dir = os.path.dirname(folder)
json_file = "rename_" + folder_name + ".json"
json_path = os.path.join(parent_dir, json_file)
name_dict = {}
csv_list = []
png_count = 0
not_in_dict = 0
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
                    new_name = filename.replace(",", "")
                    name_dict[filename] = new_name
with open(json_path, "w") as f:
    json.dump(name_dict, f)
df_list = []
for csv_path in csv_list:
    df = pd.read_csv(csv_path)
    df["seed"] = ""
    # Skip the first row as it is the prompt
    for index, row in df.iterrows():
        if index > 0:
            seeds = []
            for col in row.index:
                if isinstance(row[col], str) and row[col].startswith('"') and row[col].endswith('"'):
                    row[col] = row[col][1:-1]
                    dir, filename = os.path.split(row[col])
                    if filename.endswith(".png"):
                        csv_set.add(filename)
                        png_count += 1
                        # Check if the filename is valid and exists in the folder
                        if os.path.isfile(os.path.join(parent_dir, row[col])):
                            if filename in name_dict:
                                filename = name_dict[filename]
                                row[col] = os.path.join(dir, filename)
                            else:
                                not_in_dict += 1
                            seed = get_seed(filename)
                            seeds.append(seed)
                        else:
                            # Print an error message
                            print(f"Error: {filename} is not a valid or existing file.")
                    row[col] = os.path.join(parent_dir, row[col])
            if len(set(seeds)) == 1:
                seed = seeds[0]
                df.loc[index, "seed"] = seed
            else:
                # Print a warning with the csv file name, the row entry and the seeds
                print(f"Warning: The seeds in row {index} of {csv_path} are not consistent. The row entry is: {row}. The seeds are: {seeds}.")
    df_list.append(df)
percentage = round((not_in_dict / png_count) * 100, 2) if png_count > 0 else 0
print(f"{percentage}% of {png_count} PNG files are not in the dictionary. This means that {not_in_dict} of the PNG files in the folder have not been renamed or do not exist.")
# Use the set difference to show the PNG files that are not in the CSV files
print(f"The following PNG files are not in the CSV files: {png_set - csv_set}. This means that some of the PNG files in the folder are not referenced in the CSV files.")
print(f"There are {len(csv_list)} CSV files and {len(name_dict)} PNG files to be modified.")
print("The following files will be renamed:")
for old_name, new_name in name_dict.items():
    print(f"{old_name} -> {new_name}")
answer = input("Do you want to rename the files in the folder and save the csv files? (y/n) ")
if answer.lower() == "y":
    for root, dirs, files in os.walk(folder):
        for filename in files:
            if filename in name_dict:
                file_path = os.path.join(root, filename)
                new_name = name_dict[filename]
                os.rename(file_path, os.path.join(root, new_name))
                print(f"Renamed {filename} to {new_name}.")
    for csv_path, df in zip(csv_list, df_list):
        df.to_csv(csv_path, index=False)
        print(f"Saved {csv_path}.")
else:
    print("No files were renamed or saved.")
