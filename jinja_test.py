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

def format_for_mturk_substitution(number_of_images):
    # Format the strings for Amazon Mechanical Turk substitution
    return [f"image{i+1}" for i in range(number_of_images)]



def generate_survey_template(country_csv_file, survey_items_csv, template_html, short_instructions, full_instructions, output_folder="output_surveys"):
    country_df = pd.read_csv(country_csv_file, quotechar='"')
    survey_items_df = pd.read_csv(survey_items_csv, quotechar='"')
    country_name = get_country_name(country_csv_file)
    number_of_images = len(get_png_column_headers(country_df))
    print(f'survey_items_df: {survey_items_df}')
    print(f'country_df: {country_df}')

    # Create a Jinja2 environment
    env = Environment(loader=FileSystemLoader('.'))

    # Load the survey template HTML
    template = env.get_template(template_html)

    def make_images_block(number_of_images):
        images_block = ''
        for i in range(number_of_images):
            images_block += f"""

                                <div data-id="{i}" class="image">
                                    <span class="number">{i}</span>
                                    <img src="image{i}" alt="Image {i}">
                                </div>
                            """
        return images_block

    # Prepare container block (simplified for Jinja)
    container_block = ''
    for i, row in survey_items_df.iterrows():
        print(f'row {i}: {row}')
        images_block = make_images_block(number_of_images)
        container_block += f"""
        <div class="sortable">
            <div class="item">
              <h2>{row['Item Title']}</h2>
              <p>{row['Item Text']}</p>
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
    output_file = os.path.join(output_folder, f"{country_name}_survey.html")
    with open(output_file, "w") as file:
        file.write(rendered_template)

    return output_file

# Example usage:
country_csv_file = "m3c_eval/China/China_files.csv"
survey_items_csv = "human_survey_items.csv"
template_html = "survey_template.html"
short_instructions = "Rank the images or provide ratings as instructed."
full_instructions = "Follow the instructions carefully and provide your rankings or ratings accordingly."

output_file_path = generate_survey_template(country_csv_file, survey_items_csv, template_html, short_instructions, full_instructions)

# Print the path to the saved file
print(f"Survey template saved to: {output_file_path}")