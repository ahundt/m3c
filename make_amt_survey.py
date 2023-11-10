# Import the necessary libraries
import argparse
import pandas as pd
import os
from jinja2 import Template, Environment, FileSystemLoader
from tqdm import tqdm
from util import find_csv_files, save_and_load_dict_with_timestamp

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=str, default="m3c_eval", help="The directory containing the subfolders of images")
    parser.add_argument("--items", type=str, default="human_survey_items.csv", help="The file containing the item titles, texts, and types")
    parser.add_argument("--item_template", type=str, default="survey_template.html", help="The file containing the survey HTML template")
    parser.add_argument("--output_folder", type=str, default="output_surveys", help="The folder to save the generated surveys")
    parser.add_argument("--url_prefix", type=str, default="https://raw.githubusercontent.com/ahundt/m3c_eval/main", help="Options are: github")
    parser.add_argument("--log_folder", type=str, default="logs", help="The folder where log files are stored")
    parser.add_argument("--log_file_basename", type=str, default="log", help="The base name for the log file")
    parser.add_argument("--description", type=str, default=None, help="A description of the run for future reference")
    parser.add_argument("--resume", type=str, default=None, help="If set to a file path (str), resumes from the specified log file. If set to True (bool), loads the most recent log file. Defaults to None")
    parser.add_argument("--short_instructions", type=str, default="Drag and drop the images to rank them from most to least relevant.", help="Short instructions for the survey")
    parser.add_argument("--full_instructions", type=str, default="Drag and drop the images to rank them from most to least relevant. If you are unsure, you can skip the item.", help="Full instructions for the survey")
    return vars(parser.parse_args())


def get_png_column_headers(dataframe):
    """
    Get the headers of columns containing ".png" paths in the first row of the DataFrame.

    Parameters:
    - dataframe: pandas DataFrame

    Returns:
    - List of headers for columns containing ".png" paths
    """
    # Get the first row of the DataFrame
    first_row = dataframe.iloc[0]

    # Find columns containing ".png" paths
    png_columns = [col for col in dataframe.columns if ".png" in str(first_row[col])]

    # Return the headers for columns containing ".png" paths
    return png_columns


def get_country_name(country_csv_file):
    # Get country name from the file path
    country = os.path.splitext(os.path.basename(country_csv_file))[0].split('_')[0]
    return country


def update_image_paths(country_df, url_prefix):
    # Update paths to web addresses using --url_prefix
    png_columns = get_png_column_headers(country_df)
    for column in png_columns:
        country_df[column] = url_prefix + '/' + country_df[column]
    return country_df


def format_for_mturk_substitution(number_of_images):
    # Format the strings for Amazon Mechanical Turk substitution
    return [f"image{i}" for i in range(number_of_images)]


def generate_survey_template(country_csv_file, survey_items_csv, template_html, short_instructions, full_instructions, output_folder="output_surveys", url_prefix="https://raw.githubusercontent.com/ahundt/m3c_eval/main"):
    country_df = pd.read_csv(country_csv_file, quotechar='"')
    survey_items_df = pd.read_csv(survey_items_csv, quotechar='"')
    country_name = get_country_name(country_csv_file)
    png_column_headers = get_png_column_headers(country_df)
    number_of_images = len(png_column_headers)
    # print(f'survey_items_df: {survey_items_df}')
    # print(f'country_df: {country_df}')

    # Create a Jinja2 environment
    env = Environment(loader=FileSystemLoader('.'))

    # Load the survey template HTML
    template = env.get_template(template_html)

    def make_images_block(number_of_images):
        # Updated block to include arrows and editable text boxes
        images_block = ''
        for i in range(number_of_images):
            images_block += f"""
                <div data-id="{i}" class="image">
                    <span class="number">{i + 1}</span>
                    <input type="text" class="edit-box" name="image{i}" value="{i + 1}" />
                    <span class="arrow left">&#9664;</span>
                    <span class="arrow right">&#9654;</span>
                    <img src="${{image{i}}}" alt="Image {i}" style="width: 200px; height: auto;">
                </div>
            """
        return images_block

    # Prepare container block (simplified for Jinja)
    container_block = ''
    for i, row in survey_items_df.iterrows():
        print(f'row {i}: {row}')
        images_block = make_images_block(number_of_images)
        container_block += f"""
        <div class="sortable items-container">
            <div class="item">
              <h2>{row['Item Title']}</h2>
              <p>{row['Item Text']}</p>
              <p>Image Description: <b>${{prompt}}</b></p>
              {images_block}
              <!-- Simplified conditions for other item types if needed -->
            </div>
        </div>
        """

    # Crowd-form string substitution
    crowd_form = f"""
        <crowd-form>
            <div class="container">
                {container_block}
            </div>
            <!-- Additional Crowd HTML Elements for Mechanical Turk -->
            <short-instructions>
                <p>{short_instructions}</p>
            </short-instructions>
            <full-instructions>
                <p>{full_instructions}</p>
                <!-- Additional detailed instructions go here -->
            </full-instructions>
        </crowd-form>
    """

    # Process data and render the template
    try:
        rendered_template = template.render(
            country=country_name,
            crowd_form=crowd_form
        )
    except Exception as e:
        print(f"Jinja2 Error: {e}")
        # Handle the error accordingly, e.g., raise it again, log it, or provide a default template.

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Save the result to a file
    survey_html_path = os.path.join(output_folder, f"{country_name}_survey.html")
    with open(survey_html_path, "w") as file:
        file.write(rendered_template)

    # Process the survey CSV and save it to a file
    country_df = update_image_paths(country_df, url_prefix)
    survey_csv_path = os.path.join(output_folder, f"{country_name}_survey.csv")
    # rename all the columns to image1, image2, etc.
    [country_df.rename(columns={col: f"image{i}"}, inplace=True) for i, col in enumerate(png_column_headers)]
    # move seed to the first column
    country_df = country_df[["seed"] + [col for col in country_df.columns if col != "seed"]]

    # Save the result to a file
    country_df.to_csv(survey_csv_path, index=False)

    return survey_html_path, survey_csv_path 

def main():
    args = parse_args()

    # Load items CSV
    items_df = pd.read_csv(args["items"])

    # Process surveys for each country
    for country_csv in tqdm(find_csv_files(args["directory"]), desc="Processing Countries"):
        # Process the country survey
        survey_items_csv = args["items"]
        template_html = args["item_template"]
        survey_html_path, survey_csv_path = generate_survey_template(
            country_csv, 
            survey_items_csv, 
            template_html, 
            short_instructions=args["short_instructions"], 
            full_instructions=args["full_instructions"], 
            output_folder=args["output_folder"],
            url_prefix=args["url_prefix"])

        # Print paths to generated files
        print(f"Generated HTML file: {survey_html_path}")
        print(f"Generated CSV file: {survey_csv_path}")

if __name__ == "__main__":
    main()
