import pandas as pd
from itertools import combinations

def binary_rank_table(df, network_models):
    """
    Break down rank items into binary image comparison tasks and create a new DataFrame.

    Parameters:
        df (pandas.DataFrame): The DataFrame containing survey responses.
        network_models (list): List of network models used for evaluation.

    Returns:
        pandas.DataFrame: The new DataFrame with binary comparison tasks.
    """

    # Filter the DataFrame for "Rank" items and relevant network models
    rank_df = df[(df['Item Type'] == "Rank") & df['Neural Network Model'].isin(network_models)]

    # Define key columns
    key_columns = ["Item Title Index", "Item Title", "Item Type",
                   "Country", "Source CSV Row Index", "Input.prompt", "Input.seed", "HITId", "WorkerId"]

    # Define binary key columns for left and right sides
    binary_key_columns = ["Neural Network Model", "Image Shuffle Index"]
    left_key_columns = [f'Left {col}' if col in binary_key_columns else col for col in key_columns]
    right_key_columns = [f'Right {col}' if col in binary_key_columns else col for col in key_columns]

    # Initialize the new DataFrame for binary rank tasks
    binary_rank_df = pd.DataFrame(columns=[
        'Left Binary Rank Image', 'Right Binary Rank Image',
        'Left Neural Network Model', 'Right Neural Network Model',
        'Left Image Shuffle Index', 'Right Image Shuffle Index',
        'Binary Rank Response Left Image is Greater'] + key_columns)

    # Iterate through unique pairs of images for binary comparison
    for _, group in rank_df.groupby(['Item Title Index', 'HITId']):
        image_combinations = list(combinations(sorted(group['Image File Path'].to_list()), 2))

        for left_image, right_image in image_combinations:
            left_group = group[group['Image File Path'] == left_image]
            right_group = group[group['Image File Path'] == right_image]

            left_response = left_group['Response'].values[0]
            right_response = right_group['Response'].values[0]

            left_nn = left_group['Neural Network Model'].values[0]
            right_nn = right_group['Neural Network Model'].values[0]

            # Handle None values
            binary_response = None if None in (left_response, right_response) else left_response < right_response

            # Construct the new row for the binary rank DataFrame
            row_data = {
                'Left Binary Rank Image': left_image,
                'Right Binary Rank Image': right_image,
                'Left Neural Network Model': left_nn,
                'Right Neural Network Model': right_nn,
                'Left Image Shuffle Index': left_group['Image Shuffle Index'].values[0],
                'Right Image Shuffle Index': right_group['Image Shuffle Index'].values[0],
                'Binary Rank Response Left Image is Greater': binary_response
            }

            # Add key columns to the row
            for col in key_columns:
                row_data[col] = group[col].values[0]

            # Append the new row to the binary rank DataFrame
            binary_rank_df = pd.concat([binary_rank_df, pd.DataFrame([row_data])], ignore_index=True)

    return binary_rank_df

if __name__ == "__main__":
    # Example usage with your provided data
    data = {
        'Item Title Index': [1, 1, 2, 2, 3, 3],
        'Item Title': ['Title1', 'Title1', 'Title2', 'Title2', 'Title3', 'Title3'],
        'Item Type': ['Rank', 'Rank', 'Rank', 'Rank', 'Rank', 'Rank'],
        'Neural Network Model': ['baseline', 'contrastive', 'baseline', 'contrastive', 'baseline', 'contrastive'],
        'Image File Path': ['img1.png', 'img2.png', 'img1.png', 'img2.png', 'img1.png', 'img2.png'],
        'Image Shuffle Index': [1, 2, 1, 2, 1, 2],  # Add this line
        'Response': [2, 1, 3, 4, 2, 3],
        'HITId': ['hit1', 'hit1', 'hit2', 'hit2', 'hit3', 'hit3'],
        'Country': ['Country1', 'Country1', 'Country2', 'Country2', 'Country3', 'Country3'],
        'Source CSV Row Index': [1, 2, 3, 4, 5, 6],
        'Input.prompt': ['Prompt1', 'Prompt1', 'Prompt2', 'Prompt2', 'Prompt3', 'Prompt3'],
        'Input.seed': [123, 123, 456, 456, 789, 789],
        'WorkerId': ['Worker1', 'Worker1', 'Worker2', 'Worker2', 'Worker3', 'Worker3']
    }

    df = pd.DataFrame(data)

    network_models = ["baseline", "contrastive"]

    binary_rank_df = binary_rank_table(df, network_models)

    # Save the binary rank results to a CSV file
    binary_rank_df.to_csv('binary_rank_results.csv', index=False)
