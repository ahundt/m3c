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
    parser.add_argument("--consent_text", type=str, default="You are being asked to participate in a research study being conducted by the Bot Intelligence Group at Carnegie Mellon University. Participation is voluntary. The purpose of this study is to understand ways to better represent culture in AI - generated images.  Any reports and presentations about the findings from this study will not include your name or any other information that could identify you.")
    return vars(parser.parse_args())

country_name_to_adj = {
    'China':'Chinese',
    'India':'Indian',
    'Mexico':'Mexican',
    'Korea':'Korean',
    'Nigeria':'Nigerian'
}

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
    country_adj = country_name_to_adj[country_name]
    png_column_headers = get_png_column_headers(country_df)
    number_of_images = len(png_column_headers)
    number_of_items = len(survey_items_df)
    country_df['country'] = country_name
    country_df['country_adj'] = country_adj

    # Create a Jinja2 environment
    env = Environment(loader=FileSystemLoader('.'))

    # Load the survey template HTML
    template = env.get_template(template_html)

    def make_input_block(item_type, promptrow, number_of_images, label_text=""):
        input_block = ""
        if item_type == "Rank":
            for i in range(1, number_of_images + 1):
                input_block += f"""
                    <td>
                        <div style="text-align: center;">
                            <img src="${{img{i}}}" style="width: 25vw; max-width: 300px; max-height: 300px;"/>
                            <input type="number" id="promptrow{promptrow}-img{i}-rating" name="promptrow{promptrow}-img{i}-rating" value="" min="1" max="{number_of_images}" required>
                        </div>
                    </td>
                """
        elif item_type == "Short Answer":
            input_block += f"""
                <td colspan={number_of_images}>
                    <div style="text-align: center;">
                        <textarea id="promptrow{promptrow}-short-answer" name="promptrow{promptrow}-short-answer" rows="5" cols="40" maxlength="300" required></textarea>
                    </div>
                </td>
            """
        elif item_type == "Binary Checkbox":
            for i in range(1, number_of_images + 1):
                input_block += f"""
                    <td>
                        <div style="text-align: center;">
                            <img src="${{img{i}}}" style="width: 25vw; max-width: 300px; max-height: 300px;"/>
                            <label for="promptrow{promptrow}-img{i}-checkbox">
                                <input type="checkbox" id="promptrow{promptrow}-img{i}-checkbox" name="promptrow{promptrow}-img{i}-checkbox" value="1">
                                {label_text}
                            </label>
                        </div>
                    </td>
                """
        return input_block

    # Create container block for items within the same table cell
    container_block = ""
    for i, row in survey_items_df.iterrows():
        item_type = row["Item Type"]
        # Create a block for input (ranking or short answer)
        input_block = make_input_block(item_type, promptrow=i+1, number_of_images=number_of_images)
        # Add the description row with the input
        container_block += f"""
            <tr>
                <td style="text-align: left;" colspan={number_of_images}>
                    <h3>{row['Item Title']}</h3>
                    {f'<p>Image Description: <b>${{prompt}}</b></p>' if row['Include Description'] else ''}
                    <p style="max-width:50rem;">{row["Item Text"]}</p>
                </td>
            </tr>
            <tr>
                {input_block}
            </tr>
        """
    
    # Complete the crowd form
    crowd_form = f"""
        <crowd-form>
            <div>
                <h3>{consent_title} about Image Generators and {country_name}</h3>
                <p>{consent_text}</p>
                <!--<p>Your Mechanical Turk Worker ID will be used to distribute the payment to you, but we will not store your worker ID with your survey responses. Please be aware that your Mturk Workers ID can potentially be linked to information about you on your Amazon Public Profile page, however we will not access any personally identifying information from your Amazon Public Profile.</p>-->
                <p style="color: darkred; font-weight: bold;">By submitting answers to this survey, you are agreeing to participate in this study</p>
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


def save_as_csv_and_tex(data, output_folder, output_filename, description):
    """
    Save DataFrame as both CSV and LaTeX files.

    Parameters:
    - data (DataFrame): The DataFrame to be saved.
    - output_folder (str): The path to the output folder.
    - output_filename (str): The base name for the output files.
    - description (str): Description for the saved files.

    Returns:
    None
    """
    # Save as CSV
    csv_file_path = os.path.join(output_folder, f"{output_filename}.csv")
    data.to_csv(csv_file_path)  # Avoid saving the DataFrame index
    print(f'{description} saved as CSV to: {csv_file_path}')

    # Save as LaTeX
    tex_file_path = os.path.join(output_folder, f"{output_filename}.tex")
    data.to_latex(tex_file_path)
    print(f'{description} saved as LaTeX to: {tex_file_path}')

def survey_summary_stats(csv_files, survey_items_csv, output_folder="output_surveys", output_filename="survey_summary_stats"):
    """
    Process survey data, compute summary statistics, and save as CSV and LaTeX files.

    Parameters:
    - csv_files (list): A list of paths to the survey CSV files.
    - survey_items_csv (str): The path to the survey items CSV file.
    - output_folder (str): The path to the output folder.
    - output_filename (str): The base name for the output files.

    Returns:
    None
    """
    # Load survey items
    items = pd.read_csv(survey_items_csv)
    df_list = []

    # Load and combine survey data
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        df['Country'] = get_country_name(csv_file)
        df_list.append(df)

    df = pd.concat(df_list)

    # Calculate summary statistics
    unique_prompts = df['prompt'].nunique()
    unique_seeds = df['seed'].nunique()
    image_cols = df.filter(regex='img\d')
    unique_models = image_cols.nunique().max() if not image_cols.empty else 0
    items_per_page = items.shape[0]
    unique_pages = df.shape[0]
    unique_countries = df['Country'].nunique()

    counts = pd.DataFrame({
        'Summary': ['Counts'],
        'Unique Prompts': [unique_prompts],
        'Unique Seeds': [unique_seeds],
        'Unique Models': [unique_models],
        'Items per Page': [items_per_page],
        'Unique Pages': [unique_pages],
        'Unique Countries': [unique_countries]
    })
    # set the counts row name to "Overall Counts"
    counts.set_index('Summary', inplace=True)
    # Rename the 'Counts' column to 'Summary'
    counts.rename(columns={'Counts': 'Summary'}, inplace=True)
    # Make the Counts row named 'Summary'
    counts.index.name = 'Summary'

    # Per-country statistics
    per_country_stats = df.groupby('Country').agg({
        'prompt': 'nunique',
        'seed': 'nunique',
        'Country': 'count'
    }).rename(columns={'prompt': 'Unique Prompts Per Country', 'seed': 'Unique Images Per Model', 'Country': 'Number of Survey Pages'})

    save_as_csv_and_tex(counts, output_folder, f"{output_filename}_counts", "Counts")
    save_as_csv_and_tex(per_country_stats, output_folder, f"{output_filename}_per_country_stats", "Per Country Stats")




def main():
    args = parse_args()

    # list of survey html and csv files
    html_files = []
    csv_files = []

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
        html_files.append(survey_html_path)
        csv_files.append(survey_csv_path)

        # Print paths to generated files
        print(f"Generated HTML file: {survey_html_path}")
        print(f"Generated CSV file: {survey_csv_path}")

        # Shuffle images for each row
        country_df = pd.read_csv(survey_csv_path)
        shuffled_survey_csv_path = shuffle_images_and_save(country_df, get_country_name(country_csv), args["output_folder"])
        print(f"Generated shuffled CSV file: {shuffled_survey_csv_path}")

    # Get the summary stats for all the surveys and save them to a file
    survey_summary_stats(csv_files, args["items"], args["output_folder"])

if __name__ == "__main__":
    main()
