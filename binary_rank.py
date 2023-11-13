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

    # Initialize lists to store binary rank data
    left_images = []
    right_images = []
    binary_responses = []

    # Iterate through each unique pair of images for binary comparison
    for _, group in rank_df.groupby(['Item Title Index', 'HITId']):
        items = group['Image File Path'].tolist()

        # Generate combinations of image pairs
        image_combinations = list(combinations(items, 2))

        for left_image, right_image in image_combinations:
            left_images.append(left_image)
            right_images.append(right_image)

            left_response = group[group['Image File Path'] == left_image]['Response'].values[0]
            right_response = group[group['Image File Path'] == right_image]['Response'].values[0]

            # Handle cases with None values
            if left_response is None or right_response is None:
                binary_responses.append(None)
            else:
                binary_responses.append(left_response < right_response)

    # Create the binary rank DataFrame
    binary_rank_df = pd.DataFrame({
        'Left Binary Rank Image': left_images,
        'Right Binary Rank Image': right_images,
        'Binary Rank Response Left is Greater': binary_responses
    })

    return binary_rank_df

# Example usage with your provided data
data = {
    'Item Title Index': [1, 1, 2, 2, 3, 3],
    'Item Type': ['Rank', 'Rank', 'Rank', 'Rank', 'Rank', 'Rank'],
    'Neural Network Model': ['baseline', 'contrastive', 'baseline', 'contrastive', 'baseline', 'contrastive'],
    'Image File Path': ['img1.png', 'img2.png', 'img1.png', 'img2.png', 'img1.png', 'img2.png'],
    'Response': [2, 1, 3, 4, 2, 3],
    'HITId': ['hit1', 'hit1', 'hit2', 'hit2', 'hit3', 'hit3']
}

df = pd.DataFrame(data)

network_models = ["baseline", "contrastive"]

binary_rank_df = binary_rank_table(df, network_models)

# Save the binary rank results to a CSV file
binary_rank_df.to_csv('binary_rank_results.csv', index=False)
