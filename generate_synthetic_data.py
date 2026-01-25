
import pandas as pd
import numpy as np

# 1. Load Original Data
input_file = 'google_form_950_responses.csv'
output_file = 'synthetic_student_data.csv'

try:
    df = pd.read_csv(input_file)
except FileNotFoundError:
    print(f"Error: {input_file} not found.")
    exit()

# 2. Define Generation Logic
# We will generate synthetic data by maintaining the distribution of each column.
# Note: This simple method assumes independence between features. 
# For more complex relationships, libraries like SDV (Synthetic Data Vault) are recommended.

num_samples = 1000  # How many new rows to generate
synthetic_data = {}

print(f"Generating {num_samples} new synthetic samples...")

for col in df.columns:
    # Check if column is numeric
    if pd.api.types.is_numeric_dtype(df[col]):
        # Generate random values based on mean and std, preserving original range
        mean = df[col].mean()
        std = df[col].std()
        
        # Determine min/max to cap values
        min_val = df[col].min()
        max_val = df[col].max()
        
        # Generate normal distribution and clip
        syn_col = np.random.normal(loc=mean, scale=std, size=num_samples)
        syn_col = np.clip(syn_col, min_val, max_val)
        
        # If original was integer, round the synthetic ones
        if pd.api.types.is_integer_dtype(df[col]):
            syn_col = np.round(syn_col).astype(int)
            
        synthetic_data[col] = syn_col
        
    else:
        # For categorical, sample based on probability distribution of original data
        # Check if probability sum to 1.0 (sometimes float precision issues)
        probs = df[col].value_counts(normalize=True)
        # Normalize just in case
        probs = probs / probs.sum()
        syn_col = np.random.choice(probs.index, size=num_samples, p=probs.values)
        synthetic_data[col] = syn_col

# 3. Create DataFrame and Save
df_syn = pd.DataFrame(synthetic_data)

# Combine with original if desired, or save separately. Here we verify balance.
print("Synthetic Data Head:")
print(df_syn.head())

# Save
df_syn.to_csv(output_file, index=False)
print(f"\nSuccessfully saved synthetic data to {output_file}")
