# Import the necessary libraries
import argparse
import pandas as pd
import os
from jinja2 import Template
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
    return vars(parser.parse_args())

def process_country_survey(item_template, items_df, country_csv, output_folder, url_prefix):
    # Load country-specific file containing prompt, images, and seed
    country_df = pd.read_csv(country_csv)

    # Get country name from the file path
    country = os.path.splitext(os.path.basename(country_csv))[0].split('_')[0]

    # Update paths to web addresses using --url_prefix
    for column in country_df.columns:
        if country_df[column].dtype == 'O' and country_df[column].str.lower().str.endswith('.png').any():
            # Modify columns with PNG files
            country_df[column] = url_prefix + '/' + country_df[column]

    # Merge country-specific file with items_df
    survey_df = pd.merge(items_df, country_df, left_on="Item Title", right_on="prompt", how="inner")

    # Generate HTML and CSV files for the survey
    survey_html_path, survey_csv_path = generate_survey_html(item_template, survey_df, country, output_folder)

    return survey_html_path, survey_csv_path

def generate_survey_html(item_template, survey_df, country, output_folder):
    # Load the Jinja2 template
    template = Template(item_template)
    print(f'dataframe: {survey_df}')
    # Render the template with the provided data
    survey_html = template.render(items=survey_df.to_dict(orient="records"), country=country)

    # Create the output folder if it does not exist
    os.makedirs(output_folder, exist_ok=True)

    # Save the rendered HTML to a file
    output_file_path = os.path.join(output_folder, f"{country}_survey.html")
    with open(output_file_path, "w", encoding="utf-8") as output_file:
        output_file.write(survey_html)

    # Save the survey DataFrame to a CSV file
    csv_output_file_path = os.path.join(output_folder, f"{country}_survey.csv")
    survey_df.to_csv(csv_output_file_path, index=False)

    return output_file_path, csv_output_file_path

def main():
    args = parse_args()

    # Load items CSV
    items_df = pd.read_csv(args["items"])

    # Process surveys for each country
    for country_csv in tqdm(find_csv_files(args["directory"]), desc="Processing Countries"):
        # Process the country survey
        survey_html_path, survey_csv_path = process_country_survey(args["item_template"], items_df, country_csv, args["output_folder"], args["url_prefix"])

        # Print paths to generated files
        print(f"Generated HTML file: {survey_html_path}")
        print(f"Generated CSV file: {survey_csv_path}")

if __name__ == "__main__":
    main()
