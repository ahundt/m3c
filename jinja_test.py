from jinja2 import Environment, FileSystemLoader
import pandas as pd
import os

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

def format_for_mturk_substitution(column_headers, number_of_images):
    # Format the strings for Amazon Mechanical Turk substitution
    return [f"Image{i+1}" for i in range(number_of_images) if f"Image{i+1}" in column_headers]

def generate_survey_template(country_csv_file, survey_items_csv, template_html, url_prefix, output_folder="output_surveys"):
    # Load data from CSV files
    country_data = pd.read_csv(country_csv_file)
    survey_items_data = pd.read_csv(survey_items_csv)
    country_name = get_country_name(country_csv_file)

    # Determine the number of images
    number_of_images = len(get_png_column_headers(country_data))

    # Update image paths
    country_data = update_image_paths(country_data, url_prefix)

    # Format strings for Amazon Mechanical Turk substitution
    mturk_image_columns = format_for_mturk_substitution(country_data.columns, number_of_images)

    # Create a Jinja2 environment
    env = Environment(loader=FileSystemLoader('.'))

    # Load the survey template HTML
    template = env.get_template(template_html)

    # Process data and render the template
    rendered_template = template.render(
        country=country_name,
        items=survey_items_data.to_dict('records'),
        number_of_images=number_of_images,
        number_of_survey_items=len(survey_items_data),
        mturk_image_columns=mturk_image_columns
    )

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Save the result to a file
    output_file = os.path.join(output_folder, f"{country_name}_survey.html")
    with open(output_file, "w") as file:
        file.write(rendered_template)

    return output_file

# Example usage:
country_csv_file = "m3c_eval/China/China_files.csv"
survey_items_csv = "human_survey_items.csv"
template_html = "survey_template.html"
url_prefix = "https://raw.githubusercontent.com/ahundt/m3c_eval/main"

output_file_path = generate_survey_template(country_csv_file, survey_items_csv, template_html, url_prefix)

# Print the path to the saved file
print(f"Survey template saved to: {output_file_path}")
