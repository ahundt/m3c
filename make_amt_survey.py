# Import the necessary libraries
import argparse
import pandas as pd
import os
import random
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
    # parser.add_argument("--short_instructions", type=str, default="For each row, enter different numbers under each image to rank from best to worst.", help="Short instructions for the survey")
    parser.add_argument("--short_instructions", type=str, default="For each row, enter different numbers in the white box under each image to rank from best (1) to worst (same as the total number of images). Each entry should contain a unique rank number. The lowest number 1 is always the best outcome, such as less offensive, a more accurate description, or fewer artifacts. If you are unsure, you can skip the item by leaving it blank.", help="Full instructions for the survey")
    parser.add_argument("--full_instructions", type=str, default="For each row, enter different numbers in the white box under each image to rank from best (1) to worst (same as the total number of images). Each entry should contain a unique rank number. The lowest number 1 is always the best outcome, such as less offensive, a more accurate description, or fewer artifacts. If you are unsure, you can skip the item by leaving it blank.", help="Full instructions for the survey")
    parser.add_argument("--consent_title", type=str, default="Consent to Participate in a Research Study", help="Title for the consent form")
    parser.add_argument("--consent_text", type=str, default="You are being asked to participate in a research study being conducted by the Bot Intelligence Group at Carnegie Mellon University. Participation is voluntary. The purpose of this study is to understand ways to represent culture in AI - generated images.  Any reports and presentations about the findings from this study will not include your name or any other information that could identify you.")
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
    return [f"img{i+1}" for i in range(number_of_images)]


def generate_survey_template(
        country_csv_file, 
        survey_items_csv, 
        template_html, 
        short_instructions, 
        full_instructions, 
        output_folder="output_surveys", 
        url_prefix="https://raw.githubusercontent.com/ahundt/m3c_eval/main",
        consent_title="Consent to Participate in a Research Study",
        consent_text="You are being asked to participate in a research study being conducted by the Bot Intelligence Group at Carnegie Mellon University. Participation is voluntary. The purpose of this study is to understand ways to represent culture in AI - generated images.  Any reports and presentations about the findings from this study will not include your name or any other information that could identify you."):
    country_df = pd.read_csv(country_csv_file, quotechar='"')
    survey_items_df = pd.read_csv(survey_items_csv, quotechar='"')
    country_name = get_country_name(country_csv_file)
    png_column_headers = get_png_column_headers(country_df)
    number_of_images = len(png_column_headers)
    number_of_items = len(survey_items_df)

    # Create a Jinja2 environment
    env = Environment(loader=FileSystemLoader('.'))

    # Load the survey template HTML
    template = env.get_template(template_html)

    def make_images_and_ratings_block(number_of_images, promptrow):
        images_and_ratings_block = ""
        for i in range(1, number_of_images+1):
            images_and_ratings_block += f"""
                <td>
                    <div style="text-align: center;">
                        <img src="${{img{i}}}" style="width: 25vw; max-width: 200px; max-height: 200px;"/>
                        <input type="number" id="promptrow{promptrow}-img{i}-rating" name="promptrow{promptrow}-img{i}-rating" value="" min="1" max="{number_of_images}" required>
                    </div>
                </td>
            """
        return images_and_ratings_block

    # Create container block for images and ratings within the same table cell
    container_block = ""
    for i, row in survey_items_df.iterrows():
        # Create a combined block for images and ratings
        images_and_ratings_block = make_images_and_ratings_block(number_of_images, promptrow=i+1)
        # Add the description row with images and ratings
        container_block += f"""
            <tr>
                <td style="text-align: left;">
                    <h3>{row['Item Title']}</h3>
                    <p>Image Description: <b>${{prompt}}</b></i></p>
                    <p>{row["Item Text"]} (1=best, {number_of_images}=worst)</p>
                </td>
                {images_and_ratings_block}
            </tr>
        """
    
    # Complete the crowd form
    crowd_form = f"""
        <crowd-form>
            <div>
                <h3>{consent_title}</h3>
                <p>{consent_text}</p>
                <!--<p>Your Mechanical Turk Worker ID will be used to distribute the payment to you, but we will not store your worker ID with your survey responses. Please be aware that your Mturk Workers ID can potentially be linked to information about you on your Amazon Public Profile page, however we will not access any personally identifying information from your Amazon Public Profile.</p>-->
                <p>
                    <label for="consent" style="color:red;"><b>By submitting answers to this survey, you are agreeing to participate in this study</b></label>
                </p>
            </div>
            <div class="container">
                <table style="text-align: center; max-width: 1600px;" id="question-container">
                    {container_block}
                </table>
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
            crowd_form=crowd_form,
            number_of_images=number_of_images,
            number_of_items=number_of_items
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

    # Rename all the columns to image1, image2, etc.
    [country_df.rename(columns={col: f"img{i+1}"}, inplace=True) for i, col in enumerate(png_column_headers)]
    
    # Add workerId column expected by MTurk
    # note this is just filler, it is not the real workerId
    country_df['workerId'] = "W1"
    # country_df['assignmentId'] = "A1"

    # Save the result to a file
    country_df.to_csv(survey_csv_path, index=False)

    return survey_html_path, survey_csv_path 


def shuffle_images_and_save(country_df, country_name, output_folder, use_row_seed=True):
    """
    Shuffle the image columns in the DataFrame while maintaining reproducibility based on a seed column.
    
    Args:
        country_df (pandas.DataFrame): The DataFrame containing the survey data.
        country_name (str): The name of the country or survey.
        output_folder (str): The folder to save the shuffled survey data.
        use_row_seed (bool, optional): If True, use the 'seed' column in each row for reproducibility.
            Defaults to True.

    Returns:
        str: The path to the saved shuffled survey CSV file.
    """
    # Copy the original DataFrame
    shuffled_df = country_df.copy()
    image_columns = get_png_column_headers(country_df)
    image_column_indices = [shuffled_df.columns.get_loc(col) for col in image_columns]

    # Shuffle only the image columns, and do so using the seed column
    # The reason is for reproducibility
    for i in range(len(shuffled_df)):
        if use_row_seed:
            seed = shuffled_df.loc[i, "seed"]
            random.seed(int(seed))
        shuffled_col_indices = image_column_indices.copy()
        random.shuffle(shuffled_col_indices)
        # print(f"Shuffled columns: {shuffled_cols}")
        shuffled_data = shuffled_df.iloc[i, shuffled_col_indices]
        shuffled_df.iloc[i, image_column_indices] = shuffled_data

    # Save the shuffled DataFrame to a new CSV file
    shuffled_csv_path = os.path.join(output_folder, f"{country_name}_survey_shuffled.csv")
    shuffled_df.to_csv(shuffled_csv_path, index=False)
    return shuffled_csv_path


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
            url_prefix=args["url_prefix"],
            consent_title=args["consent_title"],
            consent_text=args["consent_text"]
        )

        # Print paths to generated files
        print(f"Generated HTML file: {survey_html_path}")
        print(f"Generated CSV file: {survey_csv_path}")

        # Shuffle images for each row
        country_df = pd.read_csv(survey_csv_path)
        shuffled_survey_csv_path = shuffle_images_and_save(country_df, get_country_name(country_csv), args["output_folder"])
        print(f"Generated shuffled CSV file: {shuffled_survey_csv_path}")

if __name__ == "__main__":
    main()
