import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_visualize(csv_file_path):
    try:
        data = pd.read_csv(csv_file_path, encoding='utf-8')
    except UnicodeDecodeError:
        data = pd.read_csv(csv_file_path, encoding='ISO-8859-1')
        
    item_summary = data.groupby('Item').sum().reset_index()
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Item', y='Total', data=item_summary)
    plt.title('Total Sales by Item')
    plt.xlabel('Item')
    plt.ylabel('Total Sales ($)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Check if the outputImages directory exists, create if it does not
    output_dir = 'outputImages'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    plt.savefig(os.path.join(output_dir, 'sales_summary.png'))
    plt.show()

if __name__ == "__main__":
    load_and_visualize('extracted_data.csv')
