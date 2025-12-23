import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv('../data/predbpm.csv')

# Inspect the data
print(df.head())
print(df.info())

# Perform the 3-way split
# 70% Train, 30% for Val/Test
df_train, df_temp = train_test_split(df, test_size=0.3, random_state=42)

# Split the remaining 30% into two equal halves (15% each of the original)
df_val, df_test = train_test_split(df_temp, test_size=0.5, random_state=42)

# Verify sizes
print(f"Total size: {len(df)}")
print(f"Train size: {len(df_train)} ({len(df_train)/len(df):.1%})")
print(f"Val size: {len(df_val)} ({len(df_val)/len(df):.1%})")
print(f"Test size: {len(df_test)} ({len(df_test)/len(df):.1%})")

# Save to CSV files
df_train.to_csv('../data/train.csv', index=False)
df_val.to_csv('../data/val.csv', index=False)
df_test.to_csv('../data/test.csv', index=False)