#Lyft Inc Dataset
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import math


lyft_dataset_df = pd.read_csv("Lyftdataset.csv",)
lyft_dataset_df.info()

def int_to_categorical(df, max_unique_values = 10):
    """ Parmeter:
    - df :  Input pd.DataFrame
    - max_unique_values: int, max number of unique values
    """

    df_copy = df.copy()
    n_rows = len(df_copy)

    for col in df_copy.select_dtypes(
        include = ['int', 'int64', 'int32']
    ).columns :
        n_unique = df_copy[col].nunique()

        if n_unique <= max_unique_values:
            df_copy[col] = df_copy[col].astype('category')

    return df_copy

# Change relevant integer columns
new_lyft_dataset_df = int_to_categorical(lyft_dataset_df)

# High-level metadata view
new_lyft_dataset_df.info()


def create_metadata(df) :
    # Initialize DataFrame object with index (row names) as feature names
    meta = pd.DataFrame(index=df.columns)
    # Extract variable types
    meta['dtype'] = df.dtypes
    # Extract number of unique values
    meta['n_unique'] = df.nunique()
    # Number of missing values 
    meta['missing_sum'] = df.isnull().sum()

# Whether feature is numeric
    meta['is_numeric'] = meta['dtype'].apply(
        lambda x: pd.api.types.is_numeric_dtype(x)
    )
    # Whether feature is categorical
    meta['is_categorical'] = meta['dtype'].apply(
        lambda x: isinstance(x, pd.CategoricalDtype) or x == object
    )
    # Sample of values (that are not missing) from dataset
    meta['sample_values'] = df.apply(lambda col: col.dropna().unique()[:5])
    # Return metadata DataFrame
    return meta

# Run the method on DataFrame and generate report
meta_df = create_metadata(new_lyft_dataset_df)

metadata = create_metadata(new_lyft_dataset_df)
print(metadata.head())  # or display(metadata) in Jupyter



# Export DataFrame to CSV file
meta_df.to_csv("metadata_report_lyft_dataset.csv")

print("Metadata report saved as 'metadata_report_lyft_dataset.csv'")


# Count, Mean, Standard Deviation, Min,
# Q1, Median, Q3, Max
summary = new_lyft_dataset_df.describe().T

# Skewness describes whether data is asymmetric (not centered)
summary["skew"] = new_lyft_dataset_df.select_dtypes(
    include = "number"
).skew()

# Kurtosis describes whether data is heavy-tailed (outliers)
summary["kurtosis"] = new_lyft_dataset_df.select_dtypes(
    include = "number"
).kurt()



# Display it
print(summary)

# Optional: Save to CSV
summary.to_csv("summary_stats_lyft_dataset.csv")
print("Summary statistics saved to 'summary_stats_lyft_dataset.csv'")



"""def plot_boxplot_grid(
    df, cols_per_row=5, figsize_per_plot=(4, 3), 
    title='Boxplots of Numeric Features'
):
    # Extract numeric features & calculate plots per row in grid
    numeric_cols = df.select_dtypes(include='number').columns
    n_cols = len(numeric_cols)
    n_rows = math.ceil(n_cols / cols_per_row)
    
    # Setup subplots
    fig, axes = plt.subplots(
        n_rows, cols_per_row, 
        figsize=(cols_per_row * figsize_per_plot[0], 
                 n_rows * figsize_per_plot[1])
    )
    axes = axes.flatten()


# Loop through each numeric feature to make a boxplot inside of grid
    for i, col in enumerate(numeric_cols):
        sns.boxplot(y=df[col], ax=axes[i])
        axes[i].set_title(col)
        axes[i].set_xlabel("")
    
    # Hide unused axes
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    plt.suptitle(title, fontsize=16, y=1.02)
    plt.show()

plot_boxplot_grid(df = new_Uber_reviews_df)
"""

def plot_boxplot_grid(
    df, cols_per_row=5, figsize_per_plot=(4, 3), 
    title='Boxplots of Numeric Features'
):
    # Filter numeric columns
    numeric_cols = df.select_dtypes(include='number').columns

    # Remove columns with all NaNs or only one unique value
    numeric_cols = [
        col for col in numeric_cols 
        if df[col].notna().sum() > 0 and df[col].nunique() > 1
    ]
    
    n_cols = len(numeric_cols)
    n_rows = math.ceil(n_cols / cols_per_row)
    
    fig, axes = plt.subplots(
        n_rows, cols_per_row, 
        figsize=(cols_per_row * figsize_per_plot[0], 
                 n_rows * figsize_per_plot[1])
    )
    axes = axes.flatten()

    for i, col in enumerate(numeric_cols):
        sns.boxplot(y=df[col], ax=axes[i])
        axes[i].set_title(col)
        axes[i].set_xlabel("")

    # Hide unused axes
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.suptitle(title, fontsize=16, y=1.02)
    plt.show()

plot_boxplot_grid(df = new_lyft_dataset_df)


def plot_violin_grid(
    df, cols_per_row=5, figsize_per_plot=(4, 3), orient = "v",
    title='Violin Plots of Numeric Features'
):
    # Extract numeric features & calculate plots per row in grid
    numeric_cols = df.select_dtypes(include='number').columns
    n_cols = len(numeric_cols)
    n_rows = math.ceil(n_cols / cols_per_row)

    # Setup subplots
    fig, axes = plt.subplots(
        n_rows, cols_per_row, 
        figsize=(cols_per_row * figsize_per_plot[0], 
                 n_rows * figsize_per_plot[1])
    )
    axes = axes.flatten()


# Loop through each numeric feature to make violin plot
    for i, col in enumerate(numeric_cols):
        if orient == 'v':
            sns.violinplot(
                y=df[col], ax=axes[i], inner='box',
                linewidth=5
            )
        else:
            sns.violinplot(
                x=df[col], ax=axes[i], inner='box',
                linewidth=5
            )
        axes[i].set_title(f"{col} (Violin)")
        axes[i].set_xlabel("")
        axes[i].set_ylabel("")



# Hide any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.suptitle(title, fontsize=16, y=1.02)
    plt.show()

plot_violin_grid(
    df = new_lyft_dataset_df
)



def plot_correlation_heatmap(
    df, method='pearson', figsize=(12, 10),
    vmin=-0.5, vmax=0.5, 
    palette='coolwarm', title='Correlation Matrix Heatmap'
):
    # Calculate correlation matrix
    corr = df.select_dtypes(include="number").corr(method=method)


 # Create the heatmap
    plt.figure(figsize=figsize)
    sns.heatmap(
        corr,
        annot=True,                # Display correlation coefficients
        fmt=".1f",                 # Format to 1 decimal place
        cmap=palette,             # Color palette
        square=True,              # Make cells square
        cbar_kws={"shrink": 0.75}, # Shrink colorbar for aesthetics
        linewidths=0.5,           # Add lines between squares
        linecolor='gray',         # Line color
        annot_kws={"size": 10},    # Annotation text size
        vmin=vmin, vmax=vmax      # Set min & max values for color shading
    )
   

 # Create the heatmap
    plt.figure(figsize=figsize)
    sns.heatmap(
        corr,
        annot=True,                # Display correlation coefficients
        fmt=".1f",                 # Format to 1 decimal place
        cmap=palette,             # Color palette
        square=True,              # Make cells square
        cbar_kws={"shrink": 0.75}, # Shrink colorbar for aesthetics
        linewidths=0.5,           # Add lines between squares
        linecolor='gray',         # Line color
        annot_kws={"size": 10},    # Annotation text size
        vmin=vmin, vmax=vmax      # Set min & max values for color shading
    )
   

    plt.title(title, fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

# Plot heatmap matrix for Pearson correlation coefficient values
plot_correlation_heatmap(
    df = new_lyft_dataset_df
)

 # Create histogram
def plot_histogram_grid(
    df, cols_per_row=5, figsize_per_plot=(4, 3), bins=30,
    title='Histograms of Numeric Features'
):
    # Select numeric columns and filter out bad ones
    numeric_cols = df.select_dtypes(include='number').columns
    numeric_cols = [
        col for col in numeric_cols 
        if df[col].notna().sum() > 0 and df[col].nunique() > 1
    ]
    
    n_cols = len(numeric_cols)
    if n_cols == 0:
        print("No valid numeric columns to plot.")
        return
    
    n_rows = math.ceil(n_cols / cols_per_row)
    
    fig, axes = plt.subplots(
        n_rows, cols_per_row,
        figsize=(cols_per_row * figsize_per_plot[0],
                 n_rows * figsize_per_plot[1])
    )
    axes = axes.flatten()

    for i, col in enumerate(numeric_cols):
        axes[i].hist(df[col].dropna(), bins=bins, color='skyblue', edgecolor='black')
        axes[i].set_title(col)
        axes[i].set_xlabel('')
        axes[i].set_ylabel('Frequency')

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.suptitle(title, fontsize=16, y=1.02)
    plt.show()


plot_histogram_grid(df=new_lyft_dataset_df)


