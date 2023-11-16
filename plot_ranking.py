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

def strip_plot_rank(data, x, y, hue, filename='plot', size=(8, 6), palette=None, file_format='png', show_plot=True):
    plt.figure(figsize=size)
    
    sns.stripplot(data=data, x=x, y=y, hue=hue, palette=country_markers, jitter=0.25,
                  size=24, edgecolor='black', linewidth=1.75)
    
    plt.xlabel('Neural Network Model', fontsize=14, fontweight='bold')
    plt.ylabel('Rank', fontsize=14, fontweight='bold')
    plt.title('Rankings by Neural Network Model', fontsize=18, fontweight='bold')
    plt.legend(loc='lower left', bbox_to_anchor=(0, 0), fontsize=10)
    plt.gca().invert_yaxis()
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)
    plt.tight_layout()
    
    for i in range(len(data[x].unique()) - 1):
        plt.axvline(i + 0.5, color='gray', linestyle='--', linewidth=1)

    output_filename_png = f'{filename}.{file_format}'
    plt.savefig(output_filename_png, format=file_format)
    
    if show_plot:
        plt.show()

def parse_args():
    parser = argparse.ArgumentParser(description='Plot rankings using strip plot')
    parser.add_argument('-i', '--input', type=str, default='mmsr_rank_results-Item-Title-Country.csv',
                        help='Input CSV filename')
    parser.add_argument('-f', '--format', type=str, default='png',
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

        data = scatter_y_data(data, 'Rank', scatter_range=0.2)

        file_name = os.path.splitext(input_filename)[0]
        strip_plot_rank(data, x='Neural Network Model', y='Rank', hue='Country',
                        filename=file_name, size=(8, 6), file_format=output_format, show_plot=show_plot)
    except pd.errors.EmptyDataError:
        print("The provided CSV file is empty.")
    except pd.errors.ParserError:
        print("Error parsing the CSV file.")

if __name__ == "__main__":
    main()
