# Import the necessary libraries
import argparse
import random
import pandas as pd
import os
import json
from jinja2 import Template
import requests # Import the requests module
import tqdm
import datetime
import glob
import shutil
import re

def find_csv_files(directory):
    """
    Recursively finds and returns a list of CSV files in the given directory.

    Parameters:
        directory (str): The directory to search for CSV files.

    Returns:
        list of str: A list of full paths to CSV files found in the directory and its subdirectories.
    """
    return [os.path.join(root, file) for root, _, files in os.walk(directory) for file in files if file.endswith(".csv")]


def find_valid_image_paths(df):
    """
    Finds and validates image file paths within a pandas DataFrame.

    Parameters:
        df (pandas.DataFrame): The DataFrame containing file paths.

    Returns:
        pandas.DataFrame: A DataFrame with boolean values indicating the validity of file paths.

    This function applies a lambda function to each entry in the DataFrame, checking if each entry:
    1. Is a string (file path).
    2. Points to an existing file using os.path.isfile().
    3. Ends with '.png' (case-insensitive).

    The resulting DataFrame contains True for valid image paths and False for invalid paths.
    """
    return df.applymap(lambda entry: isinstance(entry, str) and os.path.isfile(entry) and entry.lower().endswith('.png'))


def find_valid_images_in_csv_file_directory(directory):
    """
    Finds and returns a list of valid image file paths in the given directory.

    Parameters:
        directory (str): The directory to search for image files.

    Returns:
        list of str: A list of full paths to valid image files found in the directory and its subdirectories.
    """
    csv_files = find_csv_files(directory)

    image_files = []
    for csv_file in csv_files:
        # Load the CSV file into a DataFrame using pandas
        try:
            df = pd.read_csv(csv_file)
            # Call find_valid_image_paths on the DataFrame
            image_files.extend(find_valid_image_paths(df))
        except pd.errors.EmptyDataError:
            print(f'empty csv file: {csv_file}')
            pass  # Handle empty CSV files

    return list(set(image_files))


def extract_timestamp_from_string(input_string):
    """
    Extract the first timestamp from a string using a regular expression.

    Parameters:
        input_string (str): The input string to search for a timestamp.

    Returns:
        str or None: The extracted timestamp if found, or None if not found.
    """
    match = re.search(r'(\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2})', input_string)
    if match:
        return match.group(1)
    else:
        return None


def find_csv_files(directory):
    """
    Recursively finds and returns a list of CSV files in the given directory.

    Parameters:
        directory (str): The directory to search for CSV files.

    Returns:
        list of str: A list of full paths to CSV files found in the directory and its subdirectories.
    """
    return [os.path.join(root, file) for root, _, files in os.walk(directory) for file in files if file.endswith(".csv")]


def find_valid_image_paths(df):
    """
    Finds and validates image file paths within a pandas DataFrame.

    Parameters:
        df (pandas.DataFrame): The DataFrame containing file paths.

    Returns:
        pandas.DataFrame: A DataFrame with boolean values indicating the validity of file paths.

    This function applies a lambda function to each entry in the DataFrame, checking if each entry:
    1. Is a string (file path).
    2. Points to an existing file using os.path.isfile().
    3. Ends with '.png' (case-insensitive).

    The resulting DataFrame contains True for valid image paths and False for invalid paths.
    """
    return df.applymap(lambda entry: isinstance(entry, str) and os.path.isfile(entry) and entry.lower().endswith('.png'))


def find_valid_images_in_csv_file_directory(directory):
    """
    Finds and returns a list of valid image file paths in the given directory.

    Parameters:
        directory (str): The directory to search for image files.

    Returns:
        list of str: A list of full paths to valid image files found in the directory and its subdirectories.
    """
    csv_files = find_csv_files(directory)

    image_files = []
    for csv_file in csv_files:
        # Load the CSV file into a DataFrame using pandas
        try:
            df = pd.read_csv(csv_file)
            # Call find_valid_image_paths on the DataFrame
            image_files.extend(find_valid_image_paths(df))
        except pd.errors.EmptyDataError:
            print(f'empty csv file: {csv_file}')
            pass  # Handle empty CSV files

    return list(set(image_files))


def save_and_load_dict_with_timestamp(data_dict={}, log_folder='logs', log_file_basename='log', description=None, save=True, resume=None):
    """
    Save and load a dictionary to/from a JSON file with a timestamp and preserve argument and timestamp history. The goal is to maintain records of changes made to the code, the command line arguments, and to system state in a way that is persistent across runs.

    Parameters:
        data_dict (dict): The dictionary to save or merge. You can use the 'args' key to store argument data.
        log_folder (str): The folder where log files are stored.
        log_file_basename (str): The base name for the log file.
        description (str, optional): A description of the run for future reference.
        save (str, bool, or None): Set to True, a string path, or None to save the merged dictionary.
        resume (str, bool, or None): If set to a file path (str), resumes from the specified log file.
            If set to True (bool), loads the most recent log file. Defaults to None.

    Returns:
        Tuple (dict, str or None): A tuple containing the merged dictionary and either None if not saving or the path to the saved JSON file if saving.

    Examples:
        # To save data for the first time
        data = {'args': ['arg1', 'arg2'], 'dictionary': {'key1': 'value1', 'key2': 'value2'}}
        updated_data, new_file_path = save_and_load_dict_with_timestamp(data, log_folder='logs', log_file_basename='data', save=True, resume=None)

        # To resume from a specific log file
        resumed_data, _ = save_and_load_dict_with_timestamp({}, log_folder='logs', log_file_basename='data', save=False, resume='2023_01_05_10_15_30_log.json')

        # To load the most recent log file
        latest_data, _ = save_and_load_dict_with_timestamp({}, log_folder='logs', log_file_basename='data', save=False, resume=True)
    """
    # Create the folder if it doesn't exist
    os.makedirs(log_folder, exist_ok=True)

    # Extract or generate the timestamp
    timestamp = None
    if 'timestamp' in data_dict:
        timestamp = data_dict['timestamp']
    elif isinstance(save, str) and len(save) > 19:
        timestamp = extract_timestamp_from_string(save)
    if timestamp is None:
        timestamp = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

    # Include the description in the dictionary, if provided
    if description:
        data_dict['description'] = description

    if resume:
        # Determine the file path to load or resume from
        if isinstance(resume, str):
            load_file_path = os.path.join(log_folder, resume)
        else:
            existing_files = glob.glob(f'{log_folder}/*_{log_file_basename}.json')

            # Sort existing files based on the timestamp extracted from the filename
            existing_files.sort(key=lambda x: datetime.datetime.strptime(extract_timestamp_from_string(x), '%Y_%m_%d_%H_%M_%S'), reverse=True)
            
            load_file_path = existing_files[0] if existing_files else None

        # Load data from the selected log file, if available
        if load_file_path and os.path.exists(load_file_path):
            try:
                with open(load_file_path, 'r') as json_file:
                    existing_data = json.load(json_file)
                    existing_data['resume_args'] = data_dict['args']
                    # Append the current args to the list of previous arguments
                    if 'args' in existing_data:
                        prev_args = existing_data.get('prev_args', [])
                        prev_args.append((timestamp, data_dict['args']))
                        existing_data['prev_args'] = prev_args
            except (FileNotFoundError, json.JSONDecodeError) as e:
                print(f"Error loading log file: {str(e)}")
    else:
        existing_data = data_dict

    save_file_path = None
    # Save the merged dictionary with the timestamp
    if save is not None:
        # Generate the save_file_path based on the provided save parameter
        if save is True:
            save_file_path = os.path.join(log_folder, f"{timestamp}_{log_file_basename}.json")
        elif isinstance(save, str):
            save_file_path = save

        # Backup the existing save_file_path if it exists
        if save_file_path and os.path.exists(save_file_path):
            backup_file_path = save_file_path.replace('.json', '_bkp.json')
            shutil.copyfile(save_file_path, backup_file_path)
            with open(save_file_path, 'w') as json_file:
                json.dump(existing_data, json_file, indent=4)

    return existing_data, save_file_path