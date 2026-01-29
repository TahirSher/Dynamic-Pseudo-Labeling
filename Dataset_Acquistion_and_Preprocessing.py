
install.packages("boot")
install.packages("readr")

library(boot)
library(readr)

melanoma_data <- boot::melanoma

write_csv(melanoma_data, "melanoma.csv")

import pandas as pd

melanoma_df = pd.read_csv("/content/melanoma.csv")

print(melanoma_df.head())

import pandas as pd
import numpy as np

def selective_one_hot_encoding(input_file, output_file, patient_id_column='Patient ID'):
    """
    Perform one-hot encoding on categorical columns while excluding Patient ID column.

    Parameters:
    input_file (str): Path to input CSV file
    output_file (str): Path to output encoded CSV file
    patient_id_column (str): Name of the column to exclude from encoding
    """

    # Load the dataset
    print("Loading dataset...")
    df = pd.read_csv(input_file)

    print(f"Original shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")

    # Identify categorical columns (excluding Patient ID)
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()

    # Remove Patient ID from categorical columns if it exists
    if patient_id_column in categorical_columns:
        categorical_columns.remove(patient_id_column)

    if patient_id_column in df.columns:
        print(f"Excluding column: {patient_id_column}")

    if not categorical_columns:
        print("No categorical columns found (excluding Patient ID).")
        df.to_csv(output_file, index=False)
        return df

    print(f"\nCategorical columns to encode: {categorical_columns}")

    # Display information about each categorical column
    print("\nColumn analysis:")
    for col in categorical_columns:
        unique_vals = df[col].unique()
        print(f"  {col}: {len(unique_vals)} categories → {unique_vals}")

    # Perform one-hot encoding on selected columns only
    print("\nPerforming one-hot encoding...")
    df_encoded = df.copy()

    # Keep Patient ID and numerical columns as they are
    numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()

    # Encode each categorical column
    for col in categorical_columns:
        # Get one-hot encoding for this column
        dummies = pd.get_dummies(df[col], prefix=col, dtype=int)

        # Add encoded columns to the dataframe
        df_encoded = pd.concat([df_encoded, dummies], axis=1)

        # Remove the original categorical column
        df_encoded.drop(col, axis=1, inplace=True)

    print(f"Encoded shape: {df_encoded.shape}")
    print(f"New columns: {df_encoded.columns.tolist()}")

    # Save the encoded dataset
    df_encoded.to_csv(output_file, index=False)
    print(f"\nEncoded dataset saved to: {output_file}")

    return df_encoded

# Usage
if __name__ == "__main__":
    input_csv = "/content/Breast Cancer METABRIC.csv"
    output_csv = "breast_cancer_encoded.csv"
    patient_id_col = "Patient ID"  # Change this to match your column name

    encoded_data = selective_one_hot_encoding(input_csv, output_csv, patient_id_col)

import pandas as pd
import numpy as np

def fill_nan_with_median(input_file, output_file):
  
    df = pd.read_csv(input_file)

    print(f"Original dataset shape: {df.shape}")
    print(f"NaN values before processing:\n{df.isnull().sum()}")

    numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()

    if not numerical_columns:
        print("No numerical columns found in the dataset.")
        return df

    print(f"Numerical columns: {numerical_columns}")

    for col in numerical_columns:
        if df[col].isnull().sum() > 0:
            median_value = df[col].median()
            df[col].fillna(median_value, inplace=True)
            print(f"Filled NaN in '{col}' with median: {median_value:.4f}")

    print(f"\nNaN values after processing:\n{df.isnull().sum()}")

    df.to_csv(output_file, index=False)
    print(f"\nProcessed dataset saved to: {output_file}")

    return df

if __name__ == "__main__":
    input_csv = "/content/breast_cancer_encoded.csv"
    output_csv = "breast_cancer_no_nan.csv"

    processed_data = fill_nan_with_median(input_csv, output_csv)
