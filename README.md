AMT setup
---------

# M3C: Mechanical Crowdsourcing Code Creator

M3C is a Python tool for generating surveys to collect human preferences for image ranking tasks. It's designed to make the creation of surveys for large-scale image ranking tasks more accessible. This README provides an overview of M3C, setup instructions, usage guidelines, and additional information.

## Table of Contents
- [Overview](#overview)
- [Setup Instructions](#setup-instructions)
- [Usage Instructions](#usage-instructions)
- [Additional Information](#additional-information)
- [show_forms.py](#show_formspy)
<!-- - [Contributing](#contributing) -->
<!-- - [License](#license) -->

## Overview

M3C is a tool for generating surveys used on Mechanical Turk platforms. These surveys are designed for collecting human preferences for image ranking tasks. The tool takes a dataset of images and item descriptions, generates HTML survey templates, and processes the results collected from survey participants.

- Create image ranking surveys for your dataset.
- Collect human preferences for image similarity, cultural representation, or other criteria.
- Use the provided template and data to create surveys for various countries or contexts.
- Organize and analyze the results collected from survey participants.

## Setup Instructions

Follow these steps to set up M3C:

1. **Clone the Repository**: Clone the M3C repository to your local machine:

```
git clone https://github.com/ahundt/m3c.git
git clone https://github.com/ahundt/m3c_eval.git
cd m3c
```

2. **Install Dependencies**: Make sure you have Python installed (Python 3.6+). Install the required Python packages using pip:

Note the commands may vary depending on your individual computer setup. Installing the requirements might take a bit of searching of the web for some cases.

```
pip install -r requirements.txt
```

Alternative package install command:

```
pip3 install boto3 Flask Jinja2 pandas Requests tqdm "crowd-kit[learning]" --user --upgrade
```

3. **Configuration**: Edit the configuration in `config.py` to set your preferences and paths to datasets.

## Usage Instructions

M3C is designed to be easy to use. You need a dataset of images and item descriptions in a specific format. Here are the steps to generate surveys:

1. **Prepare Your Dataset**: Ensure your dataset follows the required format. Example datasets are available at [m3c_eval](https://github.com/ahundt/m3c_eval) for reference.

2. **Create Survey Items CSV**: Create a CSV file containing item titles, texts, and types. Check the provided sample `human_survey_items.csv` for the required format.

3. **Generate Surveys**: Run M3C to generate survey templates and CSV files. Customize your survey by modifying the HTML template (`survey_template.html`).

If you're using our default example and you've cloned like above the following should run:

```
python3 make_amt_survey --directory ../m3c_eval
```

This will generate survey csv and html files in the folder `output_surveys` here is a sample directory layout:

```
├── output_surveys
│   ├── China_survey.csv
│   ├── China_survey.html
│   ├── India_survey.csv
│   ├── India_survey.html
│   ├── Korea_survey.csv
│   ├── Korea_survey.html
│   ├── Mexico_survey.csv
│   ├── Mexico_survey.html
│   ├── Nigeria_survey.csv
│   └── Nigeria_survey.html
```

These are the files you provide to amazon mechanical turk on their website, we generate separate surveys for each country in this example.


Here is a larger example of the various options you can configure:

```
python3 make_amt_survey.py --directory your_image_data --items your_survey_items.csv --item_template survey_template.html --output_folder output_surveys --url_prefix https://your-image-server.com
```

4. **Review and Upload**: Review the generated HTML files and upload them to your preferred crowdsourcing platform, such as Amazon Mechanical Turk.

5. **Collect Data**: Participants will rank images based on the survey questions.

6. **Analyze Data**: Collect and analyze the survey results.

For additional customization and advanced usage, you can modify the provided template (`survey_template.html`) to change the appearance and behavior of the generated surveys.

## Additional Information

- **CSV Format**: M3C expects a specific CSV format for your item descriptions and image URLs. Please check the example datasets provided in the [m3c_eval](https://github.com/ahundt/m3c_eval) repository.

- **Randomized Ordering**: The order of images in each survey can be randomized for improved data quality. This is done to minimize position bias in participant rankings.

- **Crowdsourcing Platforms**: M3C is designed for crowdsourcing platforms like Amazon Mechanical Turk. Ensure you have an account and the necessary platform credits to distribute surveys.


## show_forms.py

`show_forms.py` is a Python script for viewing the generated forms without using the mechanical turk website.

It allows you to perform dynamic text substitution in HTML templates based on data from CSV files. It's particularly useful when you want to display multiple forms or templates with variable data without creating separate HTML files for each case.

### Usage

#### Prerequisites

Before using `show_forms.py`, make sure you have the following prerequisites installed:

- Python
- Flask (a Python web framework)
- pandas (for working with CSV files)

You can install Flask and pandas using pip:

```bash
pip install Flask pandas
Running the Script
Save your HTML templates and corresponding CSV files in a folder. Each HTML file should contain placeholders like ${variable} where you want the dynamic text to be inserted.
```

Here is an example if you have the setup detailed at the top:

```bash
python show_forms.py --folder ./output_surveys
```

Run show_forms.py using the following command:

```bash
python show_forms.py --folder /path/to/your/folder
```

Replace /path/to/your/folder with the path to the folder containing your HTML and CSV files.

The script will start a local web server. Open your web browser and navigate to http://127.0.0.1:5000/. You will see the HTML templates with dynamic text substituted from the CSV files.

Use the browser to navigate through the forms or templates. Once all templates are processed, the server will exit automatically.

Tips
Make sure that you have an equal number of HTML and CSV files in your specified folder. Each HTML file should correspond to a CSV file.
Use ${variable} placeholders in your HTML templates. These placeholders will be replaced with data from the CSV files.

Example

For a working example, refer to the GitHub repository where you can find sample HTML and CSV files.