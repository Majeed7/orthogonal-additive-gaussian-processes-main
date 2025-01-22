import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the Excel file
file_path = 'results/fs_real/incremental_feature_addition_rf.xlsx'
excel_data = pd.ExcelFile(file_path)

is_classification = True

sheet_names_classification = ['sonar', 'nomao', 'breast_cancer_wisconsin']
if is_classification:
    sheet_names = sheet_names_classification
    performance_metric = "Mean Absolute Percentage Error"
else:
    sheet_names = result = [item for item in excel_data.sheet_names if item not in sheet_names_classification]
    performance_metric = "Accuracy"



# Set the color palette for the plots
color_palette = sns.color_palette("Set2", 10)  # You can adjust the number of colors

# Define markers for different lines
markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', '+']

# Initialize subplots
col_no = int(np.ceil(len(excel_data.sheet_names)/3))
fig, axes = plt.subplots(3, col_no, figsize=(15 * col_no, 10) )

axes = axes.flatten()
line_styles = ['-', '--', '-.', ':', 'solid', 'dashed', 'dotted']

legends = set()

# Loop through all sheets and create a subplot for each dataset
for idx, sheet_name in enumerate(excel_data.sheet_names):
    df = pd.read_excel(excel_data, sheet_name=sheet_name)
    
    # Extract feature selectors and feature counts
    feature_selectors = df['Feature Selector']
    feature_counts = list(map(int, df.columns[1:]))  # Excluding the first column
    
    ax = axes[idx]
    
    # Plot each feature selector's performance
    for i, selector in enumerate(feature_selectors):
        performance = df.iloc[i, 1:].values  # All performance values for the method
        
        # Choose a color and marker for each method
        color = color_palette[i % len(color_palette)]
        marker = markers[i % len(markers)]
        
        # Plot with anchor (marker) and color
        ax.plot(feature_counts, performance, label=selector if selector not in legends else "", color=color, marker=marker, markersize=8, linewidth=3, linestyle=line_styles[i % len(line_styles)], alpha=0.8)

        if selector not in legends:
            legends.add(selector)

    
    # Customize the plot
    ax.set_title(f"{sheet_name} Dataset", fontsize=12)
    #ax.set_xlabel('Number of Features', fontsize=12)
    if idx%4 == 0:
        ax.set_ylabel(r"Performance $\uparrow$", fontsize=14)
    #ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_xticks(feature_counts)
    ax.set_xticklabels(feature_counts, rotation=45)
    #ax.legend(title="Feature Selectors", loc='upper left', bbox_to_anchor=(1, 1), fontsize=12)

fig.legend(loc='center', bbox_to_anchor=(0.5, 1.05), ncol=10)
fig.text(0.5, 0.04, 'Number of Features', ha='center', va='center', fontsize=14)

# Adjust layout for all subplots
plt.tight_layout()
plt.show()

fig.savefig("results/fs_real/fs_real.png", dpi=500, format='png', bbox_inches='tight')
print("done!")
