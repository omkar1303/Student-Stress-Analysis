
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


# 2. Define Generation Logic (Data Augmentation / Jittering)
# Instead of generating columns independently, we sample real rows and add noise.
# This preserves correlations (e.g., high study time -> high marks).

num_samples = 1500  # Increased sample size
print(f"Generating {num_samples} new synthetic samples using Data Augmentation...")

# numeric columns to jitter
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
# categorical columns to keep (or potentially swap)
categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

new_rows = []

for _ in range(num_samples):
    # 1. Sample a random existing student
    random_row = df.sample(n=1).iloc[0].copy()
    
    # 2. Add Noise to Numeric Columns (Jittering)
    for col in numeric_cols:
        original_val = random_row[col]
        std_dev = df[col].std()
        
        # Inject 5% to 15% noise based on standard deviation
        noise = np.random.normal(0, std_dev * 0.1) 
        new_val = original_val + noise
        
        # Cap values within realistic global min/max
        min_val = df[col].min()
        max_val = df[col].max()
        new_val = np.clip(new_val, min_val, max_val)
        
        # Round if original was integer
        if pd.api.types.is_integer_dtype(df[col]):
            new_val = int(round(new_val))
            
        random_row[col] = new_val
        
    # 3. Categorical perturbation (Optional: 5% chance to flip a category)
    # For now, we keep categorical same to maintain profile consistency
    # (e.g. keeps "Department" matched with "Degree preference" if correlated)
    
    new_rows.append(random_row)

df_syn = pd.DataFrame(new_rows)

# 3. Save
df_syn.to_csv(output_file, index=False)
print("Synthetic Data Head:")
print(df_syn.head())
print(f"\nSuccessfully saved {len(df_syn)} high-quality synthetic samples to {output_file}")

