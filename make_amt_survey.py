# Import the necessary libraries
import argparse
import boto3
import random
import pandas as pd
import os
import json
from jinja2 import Template
import requests # Import the requests module
import tqdm
import datetime
import glob


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



# Define a function to read the items file
def read_items(file):
    # Check if the file exists
    if not os.path.exists(file):
        # Raise an exception
        raise FileNotFoundError(f"File {file} not found")
    # Open the file in read mode
    with open(file, "r") as f:
        # Create a pandas data frame from the file
        df = pd.read_csv(f)
        # Create an empty list to store the items
        items = []
        # Loop through the rows in the data frame and do the following for each row:
        for index, row in df.iterrows():
            # Create a dictionary with keys "title", "text", and "type", and values from the row
            item = {"title": row["Item Title"], "text": row["Item Text"], "type": row["Item Type"]}
            # Append the dictionary to the list of items
            items.append(item)
    # Return the list of items
    return items

# Define a function to read a country's csv file
def read_country(file):
    # Check if the file exists
    if not os.path.exists(file):
        # Raise an exception
        raise FileNotFoundError(f"File {file} not found")
    # Open the file in read mode
    with open(file, "r") as f:
        # Create a pandas data frame from the file
        df = pd.read_csv(f)
        # Create an empty list to store the rows
        rows = []
        # Loop through the rows in the data frame and do the following for each row:
        for index, row in df.iterrows():
            # Create a dictionary with keys "seed" and "images", and values from the row
            row_dict = {"seed": row["seed"], "images": row.drop("seed").tolist()}
            # Append the dictionary to the list of rows
            rows.append(row_dict)
    # Return the list of rows
    return rows

def create_survey_layout(items, row, template_filename="survey_template.html"):
    # Load the survey template from a file
    # "rank_template.html", "likert_template.html"
    with open(template_filename, "r") as f:
        template = Template(f.read())
    # Render the template with the items and the row
    # Loop through the items and add the images from the row
    for item in items:
        # Get the item type
        item_type = item["type"]
        # If the item type is "Rank", add the images from the row
        if item_type == "Rank":
            # Get the prompt from the item text
            prompt = item["text"]
            # Get the image URLs from the row based on the prompt
            images = [row[prompt], row["Generic Stable Diffusion"], row["Pure Finetune"], row["Positive"], row["Contrastive"]]
            # Add the images to the item
            item["images"] = images
    # Render the template with the items and the row
    layout = template.render(items=items, title=row["prompt"], seed=row["seed"])
    # Return the HIT layout
    return layout

# Define a function to create and upload a HIT
def create_hit(hit_type_id, hit_layout, assignments, client):
    # Call the create_hit_with_hit_type method of the client object with the specified parameters
    response = client.create_hit_with_hit_type(
        HITTypeId=hit_type_id,
        HITLayoutId=hit_layout,
        MaxAssignments=assignments
    )
    # Extract the HIT ID and the HIT URL from the response and return them
    hit_id = response["HIT"]["HITId"]
    hit_url = "https://workersandbox.mturk.com/mturk/preview?groupId=" + hit_type_id
    return hit_id, hit_url


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

def upload_to_s3(directory, s3_client, s3_bucket_name, s3_url_dict={}, s3_response_dict={}):
    # Create an empty list to store the S3 object URLs of the images
    s3_urls = []
    s3_dict = {}
    if os.path.isfile(current_object_dict):
        with open(current_object_dict_path, "r") as json_file:
            s3_dict = json.load(json_file)
    images = find_valid_images_in_csv_file_directory(directory)
    # if len(images): # TODO wanted to check if there are images, then do some check
    # Loop through the images in the row and do the following for each image:
    for image_path in images:
        if image_path not in s3_url_dict:
            s3_object_key = local_image_path
            # List objects in the bucket with a prefix matching the object key
            objects_exist_response = s3.list_objects(Bucket=s3_bucket_name, Prefix=s3_object_key)
            # Check if the object exists in the list of objects
            object_exists = any(obj['Key'] == s3_object_key for obj in response.get('Contents', []))
            if not object_exists:
                try:
                    response = s3.upload_file(local_image_path, s3_bucket_name, s3_object_key)
                    s3_response_dict[image_path] = response
                    s3_url = response['ResponseMetadata']['HTTPHeaders']['location']
                    s3_url_dict[image_path] = s3_url
                except botocore.exceptions.EndpointConnectionError as e:
                    print(f"S3 Upload Connection error: {e}")
                except botocore.exceptions.ClientError as e:
                    print(f"S3 Upload An error occurred: {e}")
                except FileNotFoundError:
                    print(f"S3 Upload Local file not found: {local_image_path}")
    return s3_url_dict, s3_response_dict

def create_survey_github_and_amt(args, log, save_file_path):
    return
    

# Define a main function
def main():
    # Call the function to parse the command line arguments and store the result in a variable
    args = parse_args()

    # Initialize variables based on the command line and specified files on disk
    log, save_file_path = save_and_load_dict_with_timestamp(
        data_dict={},  # Provide your data dictionary here
        log_folder=args['log_folder'],
        log_file_basename=args['log_file_basename'],
        description=args['description'],
        resume=args['resume']
    )
    
    # load the current args from the logs
    args = logs['args']
    eval_dir = args["directory"]
    platform = 'github'
    if os.path.exists(args['credentials']):
        platform = 's3'

    if platform == 'github':
        create_survey_github_and_amt(args, log, items)

    # S3 case, the following code is buggy.
    # Call the function to read the items file and store the result in a variable
    items = read_items(args["items"])
    # Read the credentials file and get the access key and secret key
    with open(args["credentials"], "r") as f:
        df = pd.read_csv(f)
        access_key = df["Access key ID"][0]
        secret_key = df["Secret access key"][0]
    
    s3_bucket = args['bucket']
    # Create a boto3 client object for MTurk sandbox using the access key and secret key
    mturk_client = boto3.client("mturk", endpoint_url="https://mturk-requester-sandbox.us-east-1.amazonaws.com", aws_access_key_id=access_key, aws_secret_access_key=secret_key)
    # Create a boto3 client object for S3 using the access key and secret key
    s3_client = boto3.client("s3", aws_access_key_id=access_key, aws_secret_access_key=secret_key)
    s3_url_dict, s3_response_dict = upload_to_s3(eval_dir, s3_client, args["bucket"])
    log["s3_url_dict"] = s3_url_dict
    log["s3_response_dict"] = s3_response_dict
    
    # save current progress to disk, s3 files have been uploaded
    log, save_file_path = save_and_load_dict_with_timestamp(
        data_dict=log,  # Provide your data dictionary here
        log_folder=args['log_folder'],
        log_file_basename=args['log_file_basename'],
        description=args['description'],
        resume=args['resume']
    )

    # Loop through the items and do the following for each item:
    subfolders = os.listdir(eval_dir)
    for subfolder in subfolders:
        # Create a HIT type with a suitable title, description, reward, duration, and keywords, and store the result in a variable
        # Call the create_hit_type method of the client object with the specified parameters
        response = mturk_client.create_hit_type(
            Title=item["title"],
            Description="Please rate the images according to the given criteria.",
            Reward=str(0.0),
            AssignmentDurationInSeconds=30000,
            Keywords="image, rating, survey",
        )
        # Extract the HIT type ID from the response and return it
        hit_type_id = response["HITTypeId"]
        # Save the HIT type ID to the output file
        save_data(args["output"], {"hit_type_id": hit_type_id})
        # Loop through the subfolders in the directory and do the following for each subfolder:
        # Get the country name from the subfolder name
        country = subfolder.split("-")[1]
        # Get the country's csv file name
        country_file = os.path.join(args["directory"], subfolder, country + "_files.csv")
        # Call the function to read the country's csv file and store the result in a variable
        rows = read_country(country_file)
        # Loop through the rows in the csv file and do the following for each row:
        for row in rows:
            # Set the random seed by combining the user id with the seed value from the row
            random.seed("user_id" + str(row["seed"]))
            # Call the function to create a HIT layout for a rank item using the S3 object URLs and store the result in a variable
            hit_layout = create_survey_layout(
                title=item["title"],
                text=item["text"],
                images=s3_urls
            )
            # Call the function to create and upload a HIT with the HIT type ID, the HIT layout, and the number of assignments, and store the result in a variable
            hit_id, hit_url = create_hit(
                hit_type_id=hit_type_id,
                hit_layout=hit_layout,
                assignments=10,
                client=mturk_client
            )
            # Save the HIT ID and the HIT URL to the output file
            save_data(args["output"], {"hit_id": hit_id, "hit_url": hit_url})
            # Print the HIT ID and the URL of the HIT for verification
            print(f"HIT ID: {hit_id}")
            print(f"HIT URL: {hit_url}")
    # Print a message indicating that the program is done
    print(f"The program is done, with local data logs in {args["log_folder"]}.")
    
# Call the main function
if __name__ == "__main__":
    main()
