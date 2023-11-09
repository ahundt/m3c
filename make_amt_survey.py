# Import the necessary libraries
import argparse
import pandas as pd
import os
from jinja2 import Template
from tqdm import tqdm
import util

def parse_args():
    # Create an argument parser object
    parser = argparse.ArgumentParser()
    # Add arguments for the directory, the items file, the credentials file, and the output file, with default values
    parser.add_argument("--directory", type=str, default="m3c_eval", help="The directory containing the subfolders of images")
    parser.add_argument("--items", type=str, default="human_survey_items.csv", help="The file containing the item titles, texts, and types")
    parser.add_argument("--credentials", type=str, default="credentials.csv", help="The file containing the AWS access key and secret key")
    parser.add_argument("--bucket", type=str, default="m3c", help="The Amazon S3 storage bucket name to upload the data to")
    parser.add_argument("--url_prefix", type=str, default="https://raw.githubusercontent.com/ahundt/m3c_eval/main", help="Options are: github")
    parser.add_argument("--output_folder", type=str, default="output", help="The folder to save the output files")
    
    # Additional arguments for save and load function capabilities
    parser.add_argument("--log_folder", type=str, default="logs", help="The folder where log files are stored")
    parser.add_argument("--log_file_basename", type=str, default="log", help="The base name for the log file")
    parser.add_argument("--description", type=str, default=None, help="A description of the run for future reference")
    parser.add_argument("--resume", type=str, default=None, help="If set to a file path (str), resumes from the specified log file. If set to True (bool), loads the most recent log file. Defaults to None")
    
    # Parse the arguments and return them as a dictionary
    return vars(parser.parse_args())

def load_country_csv(country, directory):
    csv_path = os.path.join(directory, f"{country}_files.csv")
    try:
        df = pd.read_csv(csv_path)
        return df
    except pd.errors.EmptyDataError:
        print(f'empty csv file: {csv_path}')
        return None

def process_country_survey(country, items_df, directory, url_prefix, output_folder):
    # Load the country-specific CSV file
    country_df = load_country_csv(country, directory)
    if country_df is None:
        return

    # Modify the dataframe to be ready for Mechanical Turk
    for column in country_df.columns:
        if country_df[column].dtype == 'O' and country_df[column].str.lower().str.endswith('.png').any():
            # Modify columns with PNG files
            country_df[column] = url_prefix + '/' + country_df[column]

    # Save the modified CSV to a new location
    output_csv_path = os.path.join(output_folder, f"{country}_survey.csv")
    country_df.to_csv(output_csv_path, index=False)

    # Load the survey template
    with open('survey_template.html', 'r') as template_file:
        template_content = template_file.read()

    # Create a Jinja template from the content
    template = Template(template_content)

    # Render the template with the country-specific data
    rendered_html = template.render(title=f"Survey for {country}", items=items_df.to_dict('records'))

    # Save the rendered HTML to a new location
    output_html_path = os.path.join(output_folder, f"{country}_survey.html")
    with open(output_html_path, 'w') as html_file:
        html_file.write(rendered_html)

def main():
    # Call the function to parse the command line arguments and store the result in a variable
    args = parse_args()

    # Create the output folder if it does not exist
    os.makedirs(args['output_folder'], exist_ok=True)

    # Initialize variables based on the command line and specified files on disk
    log, save_file_path = util.save_and_load_dict_with_timestamp(
        data_dict={},  # Provide your data dictionary here
        log_folder=args['log_folder'],
        log_file_basename=args['log_file_basename'],
        description=args['description'],
        resume=args['resume']
    )

    # Load the items CSV
    items_df = pd.read_csv(args['items'])

    # Loop through each country
    for country_folder in tqdm(os.listdir(args['directory']), desc="Processing Countries"):
        country_path = os.path.join(args['directory'], country_folder)
        
        # Ensure it's a directory
        if os.path.isdir(country_path):
            # Process the country-specific survey
            process_country_survey(country_folder, items_df, country_path, args['url_prefix'], args['output_folder'])

# Call the main function
if __name__ == "__main__":
    main()
