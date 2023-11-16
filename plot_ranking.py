import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import os
import numpy as np

def scatter_y_data(data, y_column, scatter_range=0.2):
    scattered_data = data.copy()
    random_values = np.linspace(-scatter_range, scatter_range, len(scattered_data))
    np.random.shuffle(random_values)
    scattered_data[y_column] += random_values
    return scattered_data

def strip_plot_rank(data, x, y, hue, filename='plot', size=(8, 6), palette=None, file_format='png', show_plot=True):
    plt.figure(figsize=size)
    
    country_markers = {
        'China': ('#DE2910', 'o'),  
        'India': ('#FF9933', 's'),  
        'Mexico': ('#006847', '^'),  
        'Korea': ('#003478', 'D'),  
        'Nigeria': ('#008751', 'P')  
    }

    sns.stripplot(data=data, x=x, y=y, hue=hue, palette=country_markers, jitter=0.2, size=24, edgecolor='black', linewidth=1.5, marker='o')
    
    plt.xlabel('Neural Network Model', fontsize=14, fontweight='bold')
    plt.ylabel('Rank', fontsize=14, fontweight='bold')
    plt.title('Rankings by Neural Network Model', fontsize=18, fontweight='bold')
    plt.legend(loc='lower left', bbox_to_anchor=(0, 0), fontsize=10)  # Move legend to bottom left inside the chart
    plt.gca().invert_yaxis()  # Reverse the Y-axis
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)
    plt.tight_layout()
    
    # Adding separator lines between each neural network model
    for i in range(len(data[x].unique()) - 1):
        plt.axvline(i + 0.5, color='gray', linestyle='--', linewidth=1)

    output_filename_png = f'{filename}.png'
    plt.savefig(output_filename_png, format='png')
    
    output_filename_pdf = f'{filename}.pdf'
    plt.savefig(output_filename_pdf, format='pdf')
    
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

        # Scatter the y-coordinates within ±0.2 using well-spaced pseudorandom numbers
        data = scatter_y_data(data, 'Rank', scatter_range=0.1)

        file_name = os.path.splitext(input_filename)[0]  # Get filename without extension
        strip_plot_rank(data, x='Neural Network Model', y='Rank', hue='Country',
                        filename=file_name, size=(8, 6), file_format=output_format, show_plot=show_plot)
    except pd.errors.EmptyDataError:
        print("The provided CSV file is empty.")
    except pd.errors.ParserError:
        print("Error parsing the CSV file.")

if __name__ == "__main__":
    main()
