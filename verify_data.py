import pandas as pd


df = pd.read_csv("data.csv")

print("Dataframe shape:")
print(df.shape)

print("\nExact column names:")
print(df.columns.tolist())

print("\nFirst five rows:")
print(df.head())

print("\nData types:")
print(df.dtypes)

print("\nMissing-value counts:")
print(df.isna().sum())

print("\nDuplicate row count:")
print(df.duplicated().sum())

print("\nUnique values of categorical columns:")
for column in ["Campaign_Channel", "Target_Segment"]:
    print(f"{column}: {sorted(df[column].unique().tolist())}")

numeric_columns = df.select_dtypes(include="number").columns
numeric_summary = df[numeric_columns].agg(["min", "max", "mean", "std"]).T
print("\nNumeric-column summary (min, max, mean, standard deviation):")
print(numeric_summary)

print("\nCampaign_ID is unique:", df["Campaign_ID"].is_unique)
print("Clicks never exceed Impressions:", bool((df["Clicks"] < df["Impressions"]).all()))
print("Exactly 2,000 records were created:", len(df) == 2_000)
