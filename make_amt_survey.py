# Import the necessary libraries
import argparse
import boto3
import random
import pandas as pd
import os

# Define a function to parse the command line arguments
def parse_args():
    # Create an argument parser object
    parser = argparse.ArgumentParser()
    # Add arguments for the directory and the items file, with default values
    parser.add_argument("--directory", type=str, default="CCUB_eval", help="The directory containing the subfolders of images")
    parser.add_argument("--items", type=str, default="human_survey_items.csv", help="The file containing the item titles, texts, and types")
    # Parse the arguments and return them as a dictionary
    return vars(parser.parse_args())

# Define a function to read the items file
def read_items(file):
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
    # Create an empty string to store the HIT layout
    hit_layout = ""
    # Add HTML code to the HIT layout to display the title and the text
    hit_layout += f"<h1>{title}</h1>\n"
    hit_layout += f"<p>{text}</p>\n"
    # Shuffle the images using random.shuffle
    random.shuffle(images)
    # Add HTML and JavaScript code to the HIT layout to display the images in a random order and allow the user to drag and drop them to rank them
    hit_layout += """
    <style>
    .container {
      display: flex;
      flex-direction: column;
      align-items: center;
    }

    .image {
      width: 200px;
      height: 200px;
      margin: 10px;
      border: 1px solid black;
    }

    .number {
      position: absolute;
      top: 10px;
      left: 10px;
      font-size: 24px;
      font-weight: bold;
    }

    .list {
      display: flex;
      flex-wrap: wrap;
      justify-content: center;
      list-style: none;
      padding: 0;
    }

    .item {
      display: flex;
      flex-direction: column;
      align-items: center;
      margin: 10px;
      padding: 10px;
      border: 1px solid black;
      cursor: move;
    }

    .answer {
      display: none;
    }
    </style>

    <div class="container">
      <h2>Drag and drop the images to rank them from best to worst</h2>
      <ul class="list" id="list">
    """
    # Loop through the images and add HTML code to the HIT layout to display each image with a number
    for i, image in enumerate(images):
        hit_layout += f"""
        <li class="item" id="{i+1}">
          <span class="number">{i+1}</span>
          <img class="image" src="{image}" alt="Image {i+1}">
        </li>
        """
    # Add HTML and JavaScript code to the HIT layout to enable the drag and drop functionality and provide an ordered list of numbers as the answer
    hit_layout += """
      </ul>
      <input class="answer" type="text" name="answer" id="answer" value="">
    </div>

    <script>
    // Get the list element
    var list = document.getElementById("list");
    // Get the answer element
    var answer = document.getElementById("answer");
    // Create a new sortable object from the list element
    var sortable = new Sortable(list, {
      // Enable drag and drop
      sort: true,
      // Update the answer value when the order changes
      onUpdate: function (evt) {
        // Get the list items
        var items = list.getElementsByTagName("li");
        // Create an empty array to store the numbers
        var numbers = [];
        // Loop through the items and get the numbers
        for (var i = 0; i < items.length; i++) {
          var number = items[i].id;
          numbers.push(number);
        }
        // Join the numbers with commas and set the answer value
        answer.value = numbers.join(",");
      },
    });
    </script>
    """
    # Return the HIT layout
    return hit_layout

# Define a function to create a HIT layout for a likert item
def create_likert_layout(title, text, images):
    # Create an empty string to store the HIT layout
    hit_layout = ""
    # Add HTML code to the HIT layout to display the title and the text
    hit_layout += f"<h1>{title}</h1>\n"
    hit_layout += f"<p>{text}</p>\n"
    # Add HTML and JavaScript code to the HIT layout to display the images and a 7-point likert scale for each image, from strongly disagree to strongly agree
    hit_layout += """
    <style>
    .container {
      display: flex;
      flex-direction: column;
      align-items: center;
    }

    .image {
      width: 200px;
      height: 200px;
      margin: 10px;
      border: 1px solid black;
    }

    .number {
      position: absolute;
      top: 10px;
      left: 10px;
      font-size: 24px;
      font-weight: bold;
    }

    .table {
      display: table;
      border-collapse: collapse;
      margin: 10px;
    }

    .row {
      display: table-row;
    }

    .cell {
      display: table-cell;
      padding: 5px;
      border: 1px solid black;
      text-align: center;
    }

    .radio {
      display: none;
    }

    .label {
      display: block;
      width: 100%;
      height: 100%;
      cursor: pointer;
    }

    .label:hover {
      background-color: lightgray;
    }

    .radio:checked + .label {
      background-color: gray;
    }

    .answer {
      display: none;
    }
    </style>

    <div class="container">
      <h2>Select a rating for each image from 1 (strongly disagree) to 7 (strongly agree)</h2>
      <div class="table">
        <div class="row">
          <div class="cell"></div>
          <div class="cell">1</div>
          <div class="cell">2</div>
          <div class="cell">3</div>
          <div class="cell">4</div>
          <div class="cell">5</div>
          <div class="cell">6</div>
          <div class="cell">7</div>
        </div>
    """
    # Loop through the images and add HTML code to the HIT layout to display each image with a number and a likert scale
    for i, image in enumerate(images):
        hit_layout += f"""
        <div class="row">
          <div class="cell"><img class="image" src="{image}" alt="Image {i+1}"><span class="number">{i+1}</span></div>
        """
        for j in range(1, 8):
            hit_layout += f"""
            <div class="cell">
              <input class="radio" type="radio" name="rating{i+1}" id="rating{i+1}{j}" value="{j}">
              <label class="label" for="rating{i+1}{j}"></label>
            </div>
            """
        hit_layout += """
        </div>
        """
    # Add HTML and JavaScript code to
   # Add HTML and JavaScript code to the HIT layout to provide an ordered list of numbers as the answer
    hit_layout += """
        <input class="answer" type="text" name="answer" id="answer" value="">
      </div>
    </div>

    <script>
    // Get the answer element
    var answer = document.getElementById("answer");
    // Get the number of images
    var n = {len(images)};
    // Create an empty array to store the ratings
    var ratings = [];
    // Loop through the images and get the ratings
    for (var i = 0; i < n; i++) {
      // Get the radio buttons for each image
      var radios = document.getElementsByName("rating" + (i + 1));
      // Loop through the radio buttons and check which one is selected
      for (var j = 0; j < radios.length; j++) {
        if (radios[j].checked) {
          // Add the rating to the array
          ratings.push(radios[j].value);
          // Break the loop
          break;
        }
      }
    }
    // Join the ratings with commas and set the answer value
    answer.value = ratings.join(",");
    </script>
    """
    # Return the HIT layout
    return hit_layout

# Define a function to create a HIT type
def create_hit_type(title, description, reward, duration, keywords):
    # Create a boto3 client object for MTurk sandbox
    client = boto3.client("mturk", endpoint_url="https://mturk-requester-sandbox.us-east-1.amazonaws.com")
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
def create_hit(hit_type_id, hit_layout, assignments):
    # Create a boto3 client object for MTurk sandbox
    client = boto3.client("mturk", endpoint_url="https://mturk-requester-sandbox.us-east-1.amazonaws.com")
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

# Define a main function
def main():
    # Call the function to parse the command line arguments and store the result in a variable
    args = parse_args()
    # Call the function to read the items file and store the result in a variable
    items = read_items(args["items"])
    # Loop through the items and do the following for each item:
    for item in items:
        # Create a HIT type with a suitable title, description, reward, duration, and keywords, and store the result in a variable
        hit_type_id = create_hit_type(
            title=item["title"],
            description="Please rate the images according to the given criteria.",
            reward=0.1,
            duration=300,
            keywords="image, rating, survey"
        )
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
                    hit_layout = create_rank_layout(
                        title=item["title"],
                        text=item["text"],
                        images=row["images"]
                    )
                # If the item type is likert, call the function to create a HIT layout for a likert item and store the result in a variable
                elif item["type"] == "Likert":
                    hit_layout = create_likert_layout(
                        title=item["title"],
                        text=item["text"],
                        images=row["images"]
                    )
                # Call the function to create and upload a HIT with the HIT type ID, the HIT layout, and the number of assignments, and store the result in a variable
                hit_id, hit_url = create_hit(
                    hit_type_id=hit_type_id,
                    hit_layout=hit_layout,
                    assignments=10
                )
                # Print the HIT ID and the URL of the HIT for verification
                print(f"HIT ID: {hit_id}")
                print(f"HIT URL: {hit_url}")
    # Print a message indicating that the program is done
    print("The program is done.")
    
# Call the main function
if __name__ == "__main__":
    main()

