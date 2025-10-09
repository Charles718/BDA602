#json_Lyft_dataset
import pandas as pd
import json

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



# Step 1: Load your CSV
Lyftdataset_df = pd.read_csv("Lyftdataset.csv")

# Change relevant integer columns
Lyftdataset_df_new = int_to_categorical(Lyftdataset_df)



# Step 2: Extract dtypes into a dictionary
dtypes = Lyftdataset_df_new.dtypes.apply(lambda dt: dt.name).to_dict()

# Optional: convert pandas types to more universal/JSON-friendly strings
# For example, 'int64' ➜ 'int64', 'float64' ➜ 'float64', etc.
# You can skip this if you're happy with how they are

# Step 3: Create metadata dictionary
metadata = {
    "dtypes": dtypes,
    "columns": list(Lyftdataset_df_new.columns),
    "n_rows": len(Lyftdataset_df_new)
}

# Step 4: Save to JSON
with open("Lyftdataset_metadata.json", "w") as f:
    json.dump(metadata, f, indent=4)

print("✅ Metadata saved as JSON!")

