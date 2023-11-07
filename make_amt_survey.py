# Import the necessary libraries
import argparse
import boto3
import random
import pandas as pd
import os
import json
from jinja2 import Template
import requests # Import the requests module

# Define a function to parse the command line arguments
def parse_args():
    # Create an argument parser object
    parser = argparse.ArgumentParser()
    # Add arguments for the directory, the items file, the credentials file, and the output file, with default values
    parser.add_argument("--directory", type=str, default="CCUB_eval", help="The directory containing the subfolders of images")
    parser.add_argument("--items", type=str, default="human_survey_items.csv", help="The file containing the item titles, texts, and types")
    parser.add_argument("--credentials", type=str, default="credentials.csv", help="The file containing the AWS access key and secret key")
    parser.add_argument("--output", type=str, default="output.json", help="The file to save the data and return values from the Amazon connection")
    # Parse the arguments and return them as a dictionary
    return vars(parser.parse_args())

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
        # Skip the header row
        df = df.iloc[1:]
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

# Define a function to create a HIT layout for a rank item
def create_rank_layout(title, text, images):
    # Load the rank template from a file
    with open("rank_template.html", "r") as f:
        template = Template(f.read())
    # Render the template with the given parameters
    hit_layout = template.render(title=title, text=text, images=images)
    # Return the HIT layout
    return hit_layout

# Define a function to create a HIT layout for a likert item
def create_likert_layout(title, text, images):
    # Load the likert template from a file
    with open("likert_template.html", "r") as f:
        template = Template(f.read())
    # Render the template with the given parameters
    hit_layout = template.render(title=title, text=text, images=images)
    # Return the HIT layout
    return hit_layout

# Define a function to create a HIT type
def create_hit_type(title, description, reward, duration, keywords, client):
    # Call the create_hit_type method of the client object with the specified parameters
    response = client.create_hit_type(
        Title=title,
        Description=description,
        Reward=str(reward),
        AssignmentDurationInSeconds=duration,
        Keywords=keywords
    )
    # Extract the HIT type ID from the response and return it
    hit_type_id = response["HITTypeId"]
    return hit_type_id

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

# Define a function to save the data and return values to a local file
def save_data(output, data):
    # Open the output file in append mode
    with open(output, "a") as f:
        # Write the data as a JSON string to the file
        f.write(json.dumps(data) + "\n")

# Define a function to upload an image to an S3 bucket and return the object URL
def upload_image(url, bucket, key, client):
    # Get the raw data of the image from the URL
    r = requests.get(url, stream=True)
    # Upload the file-like object to the S3 bucket
    client.upload_fileobj(r.raw, bucket, key)
    # Return the object URL
    return f"https://{bucket}.s3.amazonaws.com/{key}"

# Define a main function
def main():
    # Call the function to parse the command line arguments and store the result in a variable
    args = parse_args()
    # Call the function to read the items file and store the result in a variable
    items = read_items(args["items"])
    # Read the credentials file and get the access key and secret key
    with open(args["credentials"], "r") as f:
        df = pd.read_csv(f)
        access_key = df["Access key ID"][0]
        secret_key = df["Secret access key"][0]
    # Create a boto3 client object for MTurk sandbox using the access key and secret key
    mturk_client = boto3.client("mturk", endpoint_url="https://mturk-requester-sandbox.us-east-1.amazonaws.com", aws_access_key_id=access_key, aws_secret_access_key=secret_key)
    # Create a boto3 client object for S3 using the access key and secret key
    s3_client = boto3.client("s3", aws_access_key_id=access_key, aws_secret_access_key=secret_key)
    # Loop through the items and do the following for each item:
    for item in items:
        # Create a HIT type with a suitable title, description, reward, duration, and keywords, and store the result in a variable
        hit_type_id = create_hit_type(
            title=item["title"],
            description="Please rate the images according to the given criteria.",
            reward=0.1,
            duration=300,
            keywords="image, rating, survey",
            client=mturk_client
        )
        # Save the HIT type ID to the output file
        save_data(args["output"], {"hit_type_id": hit_type_id})
        # Loop through the subfolders in the directory and do the following for each subfolder:
        for subfolder in os.listdir(args["directory"]):
            # Skip the items file
            if subfolder == args["items"]:
                continue
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
                # If the item type is rank, call the function to create a HIT layout for a rank item and store the result in a variable
                if item["type"] == "Rank":
                    # Create an empty list to store the S3 object URLs of the images
                    s3_urls = []
                    # Loop through the images in the row and do the following for each image:
                    for image in row["images"]:
                        # Get the image URL from the local folder
                        image_url = os.path.join(args["directory"], subfolder, image)
                        # Get the image file name from the URL
                        image_file = os.path.basename(image_url)
                        # Upload the image to the S3 bucket and get the object URL
                        s3_url = upload_image(image_url, "my-bucket", image_file, s3_client)
                        # Append the object URL to the list of S3 object URLs
                        s3_urls.append(s3_url)
                    # Call the function to create a HIT layout for a rank item using the S3 object URLs and store the result in a variable
                    hit_layout = create_rank_layout(
                        title=item["title"],
                        text=item["text"],
                        images=s3_urls
                    )
                # If the item type is likert, call the function to create a HIT layout for a likert item and store the result in a variable
                elif item["type"] == "Likert":
                    # Create an empty list to store the S3 object URLs of the images
                    s3_urls = []
                    # Loop through the images in the row and do the following for each image:
                    for image in row["images"]:
                        # Get the image URL from the local folder
                        image_url = os.path.join(args["directory"], subfolder, image)
                        # Get the image file name from the URL
                        image_file = os.path.basename(image_url)
                        # Upload the image to the S3 bucket and get the object URL
                        s3_url = upload_image(image_url, "my-bucket", image_file, s3_client)
                        # Append the object URL to the list of S3 object URLs
                        s3_urls.append(s3_url)
                    # Call the function to create a HIT layout for a likert item using the S3 object URLs and store the result in a variable
                    hit_layout = create_likert_layout(
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
    print("The program is done.")
    
# Call the main function
if __name__ == "__main__":
    main()
