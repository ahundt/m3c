AMT setup
---------

# M3C: Mechanical Crowdsourcing Code Creator

M3C is a Python tool for generating surveys to collect human preferences for image ranking tasks. It's designed to make the creation of surveys for large-scale image ranking tasks more accessible. This README provides an overview of M3C, setup instructions, usage guidelines, and additional information.

## Table of Contents
- [Overview](#overview)
- [Setup Instructions](#setup-instructions)
- [Usage Instructions](#usage-instructions)
- [Additional Information](#additional-information)
- [Contributing](#contributing)
- [License](#license)

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

```
pip install -r requirements.txt
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

This will generate survey csv and html files in the folder `output_folder`.

These are the files you provide to amazon mechanical turk on their website.


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
