import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import os
import numpy as np

country_markers = {
    'China': '#DE2910',
    'India': '#FF9933',
    'Mexico': '#006847',
    'Korea': '#003478',
    'Nigeria': '#008751',
}

markers = {
    'China': 'o',
    'India': 's',
    'Mexico': '^',
    'Korea': 'D',
    'Nigeria': 'P',
}

def scatter_y_data(data, y_column, scatter_range=0.25):
    scattered_data = data.copy()
    # scattered_data[y_column] += np.random.uniform(-scatter_range, scatter_range, size=len(scattered_data))
    # adjust by linearly spacing the values with numpy
    scattered_data[y_column] += np.linspace(-scatter_range, scatter_range, len(scattered_data))
    return scattered_data

def strip_plot_rank(data, x='Neural Network Model', y='Rank', hue='Country', filename='plot', size=(8, 6), palette=None, file_formats=['pdf','png'], scatter=0.25, show_plot=False, title='Rankings by Neural Network Model'):
    """
    Creates a strip plot (like a one axis, multiple category scatter plot) of the given data.

    Parameters:
        data (DataFrame): The data to plot.
        x (str, optional): The column in `data` to use for the x-axis. Defaults to 'Neural Network Model'.
        y (str, optional): The column in `data` to use for the y-axis. Defaults to 'Rank'.
        hue (str, optional): The column in `data` to use for color encoding. Defaults to 'Country'.
        filename (str, optional): The name of the output file. Defaults to 'plot'.
        size (tuple, optional): The size of the plot. Defaults to (8, 6).
        palette (dict, optional): A mapping from hue levels to matplotlib colors. Defaults to None.
        file_formats (list, optional): The file formats to save the plot in. Defaults to ['.pdf','.png'].
        scatter (float, optional): The range to scatter the y-axis data. Defaults to 0.25. Set to None to disable.
        show_plot (bool, optional): Whether to display the plot. Defaults to False.

    Returns:
    None
    """
    plt.figure(figsize=size)
    data_copy = data.copy()
    if scatter is not None and scatter:
        data_copy = scatter_y_data(data_copy, y, scatter_range=scatter)
    sns.stripplot(data=data_copy, x=x, y=y, hue=hue, palette=palette, jitter=0.25,
                  size=24, edgecolor='black', linewidth=1.75)
    
    plt.xlabel('Neural Network Model', fontsize=14, fontweight='bold')
    plt.ylabel('Rank', fontsize=14, fontweight='bold')
    plt.title(title, fontsize=18, fontweight='bold')
    plt.legend(loc='lower left', bbox_to_anchor=(0, 0), fontsize=10)
    plt.gca().invert_yaxis()
    plt.yticks(np.arange(1, data_copy[y].max() + 1, 1), fontsize=12)  # Setting y-axis ticks to integers only
    plt.xticks(fontsize=12)
    plt.tight_layout()
    
    for i in range(len(data_copy[x].unique()) - 1):
        plt.axvline(i + 0.5, color='gray', linestyle='--', linewidth=0.5)
    for i in range(int(max(data_copy[y].unique()))):
        plt.axhline(i + 0.5, color='gray', linestyle='--', linewidth=0.5)

    for format in file_formats:
        output_filename = f'{filename}.{format}'
        plt.savefig(output_filename, format=format)
    
    if show_plot:
        plt.show()

def parse_args():
    parser = argparse.ArgumentParser(description='Plot rankings using strip plot')
    parser.add_argument('-i', '--input', type=str, default='mmsr_rank_results-Item-Title-Country.csv',
                        help='Input CSV filename')
    parser.add_argument('-f', '--format', type=str, default=['pdf','png'],
                        help='Output file format')
    parser.add_argument('-s', '--no-show', dest='show', action='store_false', help='Do not display the plot')
    return parser.parse_args()

def main():
    args = parse_args()
    input_filename = args.input
    output_format = args.format
    show_plot = args.show if args.show is not None else True

    if not os.path.exists(input_filename):
        print(f"File '{input_filename}' not found.")
        return

    try:
        data = pd.read_csv(input_filename)
        required_columns = {'Neural Network Model', 'Rank', 'Country'}
        if not required_columns.issubset(set(data.columns)):
            print("Required columns are missing in the CSV file.")
            return

        # data = scatter_y_data(data, 'Rank', scatter_range=0.2)

        file_name = os.path.splitext(input_filename)[0]
        strip_plot_rank(data, x='Neural Network Model', y='Rank', hue='Country',
                        filename=file_name, size=(8, 6), file_formats=output_format, show_plot=show_plot, palette=country_markers, scatter=0.25)
    except pd.errors.EmptyDataError:
        print("The provided CSV file is empty.")
    except pd.errors.ParserError:
        print("Error parsing the CSV file.")

if __name__ == "__main__":
    main()
