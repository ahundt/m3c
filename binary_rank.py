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
    left_nn_models = []
    right_nn_models = []
    left_image_indices = []
    right_image_indices = []
    binary_responses = []

    # Define key columns
    key_columns = ["Item Title Index", "Item Title", "Item Type",
                   "Country", "Source CSV Row Index", "Input.prompt", "Input.seed", "HITId", "WorkerId"]

    # Create lists for left and right key columns, which will need to both be added 
    # to the new binary_rank_df DataFrame with Left and Right prefixes
    binary_key_columns = ["Neural Network Model", "Image Shuffle Index"]
    left_key_columns = ["Left " + col if col in binary_key_columns else col for col in key_columns]
    right_key_columns = ["Right " + col if col in binary_key_columns else col for col in key_columns]

    # Iterate through each unique pair of images for binary comparison
    for _, group in rank_df.groupby(['Item Title Index', 'HITId']):
        items = group['Image File Path'].tolist()
        nn_models = group['Neural Network Model'].tolist()
        image_indices = group['Image Shuffle Index'].tolist()

        # Generate combinations of image pairs
        image_combinations = list(combinations(enumerate(items), 2))

        for (left_index, left_image), (right_index, right_image) in image_combinations:
            left_images.append(left_image)
            right_images.append(right_image)
            left_nn_models.append(nn_models[left_index])
            right_nn_models.append(nn_models[right_index])
            left_image_indices.append(image_indices[left_index])
            right_image_indices.append(image_indices[right_index])

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
        'Left Neural Network Model': left_nn_models,
        'Right Neural Network Model': right_nn_models,
        'Left Image Shuffle Index': left_image_indices,
        'Right Image Shuffle Index': right_image_indices,
        'Binary Rank Response Left is Greater': binary_responses
    })

    # Add key columns
    for col in key_columns:
        binary_rank_df[col] = rank_df.groupby(['Item Title Index', 'HITId']).head(1)[col].reset_index(drop=True)

    return binary_rank_df

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
