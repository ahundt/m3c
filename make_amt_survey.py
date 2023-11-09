# Import the necessary libraries
import argparse
import boto3
import random
import pandas as pd
import os
import json
from jinja2 import Template
import requests  # Import the requests module
import tqdm
import datetime
import glob
import shutil  # Import the shutil module for file operations
from util import save_and_load_dict_with_timestamp, find_valid_images_in_csv_file_directory


def parse_args():
    # Create an argument parser object
    parser = argparse.ArgumentParser()
    # Add arguments for the directory, the items file, the credentials file, and the output file, with default values
    parser.add_argument("--directory", type=str, default="m3c_eval", help="The directory containing the subfolders of images")
    parser.add_argument("--items", type=str, default="human_survey_items.csv", help="The file containing the item titles, texts, and types")
    parser.add_argument("--credentials", type=str, default="credentials.csv", help="The file containing the AWS access key and secret key")
    parser.add_argument("--bucket", type=str, default="m3c", help="The Amazon S3 storage bucket name to upload the data to")
    parser.add_argument("--url_prefix", type=str, default="https://raw.githubusercontent.com/ahundt/m3c_eval/main", help="Options are: github")

    # Additional arguments for save and load function capabilities
    parser.add_argument("--log_folder", type=str, default="logs", help="The folder where log files are stored")
    parser.add_argument("--log_file_basename", type=str, default="log", help="The base name for the log file")
    parser.add_argument("--description", type=str, default=None, help="A description of the run for future reference")
    parser.add_argument("--resume", type=str, default=None, help="If set to a file path (str), resumes from the specified log file. If set to True (bool), loads the most recent log file. Defaults to None")

    # Parse the arguments and return them as a dictionary
    return vars(parser.parse_args())


# Define a main function
def main():
    # Call the function to parse the command line arguments and store the result in a variable
    args = parse_args()

    # Initialize variables based on the command line and specified files on disk
    log, save_file_path = save_and_load_dict_with_timestamp(
        data_dict={'args': args},  # Provide your data dictionary here
        log_folder=args['log_folder'],
        log_file_basename=args['log_file_basename'],
        description=args['description'],
        resume=args['resume']
    )

    # Load the current args from the logs
    args = log['args']
    eval_dir = args["directory"]

    # Call the main function to generate AMT surveys
    # You can add your survey generation logic here
    # For now, I'll print the directory and items file to demonstrate the flow
    print(f"Evaluation Directory: {eval_dir}")
    print(f"Items File: {args['items']}")


# Call the main function
if __name__ == "__main__":
    main()
