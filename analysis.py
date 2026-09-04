import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import joblib
import os

# ---------------------------------------
# CREATE OUTPUT FOLDERS
# ---------------------------------------

os.makedirs("output", exist_ok=True)
os.makedirs("models", exist_ok=True)


# ---------------------------------------
# STEP 1: LOAD DATASET
# ---------------------------------------

print("Loading dataset...")

import os

print("Current working directory:")
print(os.getcwd())

print("\nFiles in current folder:")
print(os.listdir("."))

print("\nFiles in data folder:")
if os.path.exists("data"):
    print(os.listdir("data"))
else:
    print("ERROR: data folder does not exist!")

df = pd.read_excel("data/ecommerce_data.xlsx", engine="openpyxl")

print("Dataset loaded successfully!")

print("\nFirst 5 rows:")
print(df.head())


# ---------------------------------------
# STEP 2: DATA INFORMATION
# ---------------------------------------

print("\nDataset Information:")
df.info()

print("\nMissing Values:")
print(df.isnull().sum())


# ---------------------------------------
# STEP 3: DATA CLEANING
# ---------------------------------------

print("\nCleaning data...")

# Remove rows with missing Customer IDs
df = df.dropna(subset=["CustomerID"])

# Remove duplicate rows
df = df.drop_duplicates()

# Remove cancelled orders
df = df[
    ~df["InvoiceNo"]
    .astype(str)
    .str.startswith("C")
]

# Remove invalid quantities
df = df[df["Quantity"] > 0]

# Remove invalid prices
df = df[df["UnitPrice"] > 0]

print("Data cleaning completed!")

print("Cleaned dataset shape:")
print(df.shape)


# ---------------------------------------
# STEP 4: CREATE TOTAL AMOUNT
# ---------------------------------------

df["TotalAmount"] = df["Quantity"] * df["UnitPrice"]

print("\nTotalAmount column created!")


# ---------------------------------------
# STEP 5: CONVERT DATE
# ---------------------------------------

df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

print("InvoiceDate converted successfully!")


# ---------------------------------------
# STEP 6: CUSTOMER BEHAVIOR ANALYSIS
# ---------------------------------------

customer_behavior = (
    df.groupby("CustomerID")
    .agg(
        Total_Spending=("TotalAmount", "sum"),
        Total_Orders=("InvoiceNo", "nunique"),
        Total_Products=("Quantity", "sum")
    )
    .reset_index()
)

customer_behavior["Average_Order_Value"] = (
    customer_behavior["Total_Spending"]
    / customer_behavior["Total_Orders"]
)

print("\nCustomer Behavior Analysis:")
print(customer_behavior.head())


# ---------------------------------------
# STEP 7: RFM ANALYSIS
# ---------------------------------------

latest_date = df["InvoiceDate"].max()

rfm = (
    df.groupby("CustomerID")
    .agg(
        Recency=(
            "InvoiceDate",
            lambda x: (latest_date - x.max()).days
        ),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("TotalAmount", "sum")
    )
    .reset_index()
)

print("\nRFM Analysis:")
print(rfm.head())


# ---------------------------------------
# STEP 8: FEATURE SCALING
# ---------------------------------------

features = rfm[
    ["Recency", "Frequency", "Monetary"]
]

scaler = StandardScaler()

scaled_features = scaler.fit_transform(features)

print("\nFeature scaling completed!")


# ---------------------------------------
# STEP 9: K-MEANS CLUSTERING
# ---------------------------------------

kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)

rfm["Cluster"] = kmeans.fit_predict(
    scaled_features
)

print("\nCustomer clustering completed!")


# ---------------------------------------
# STEP 10: CLUSTER SUMMARY
# ---------------------------------------

cluster_summary = (
    rfm.groupby("Cluster")[
        ["Recency", "Frequency", "Monetary"]
    ]
    .mean()
)

print("\nCluster Summary:")
print(cluster_summary)


# ---------------------------------------
# STEP 11: AUTOMATIC CLUSTER NAMING
# ---------------------------------------

cluster_order = (
    rfm.groupby("Cluster")["Monetary"]
    .mean()
    .sort_values()
    .index
)

segment_mapping = {
    cluster_order[0]: "Low Value Customers",
    cluster_order[1]: "Occasional Customers",
    cluster_order[2]: "Regular Customers",
    cluster_order[3]: "VIP Customers"
}

rfm["Customer_Segment"] = (
    rfm["Cluster"].map(segment_mapping)
)

print("\nCustomer Segments:")
print(
    rfm["Customer_Segment"]
    .value_counts()
)


# ---------------------------------------
# STEP 12: MERGE CUSTOMER DATA
# ---------------------------------------

final_data = customer_behavior.merge(
    rfm,
    on="CustomerID"
)

print("\nFinal Customer Data:")
print(final_data.head())


# ---------------------------------------
# STEP 13: SAVE DATA
# ---------------------------------------


print("\nSaving output files...")

customer_segments_path = "output/customer_segments.csv"
cleaned_data_path = "output/cleaned_ecommerce_data.csv"

try:
    final_data.to_csv(
        customer_segments_path,
        index=False
    )

    df.to_csv(
        cleaned_data_path,
        index=False
    )

    print("\nFiles saved successfully!")
    print(f"Customer segments saved to: {customer_segments_path}")
    print(f"Cleaned data saved to: {cleaned_data_path}")

except PermissionError:
    print("\nERROR: Permission denied!")
    print("Please close customer_segments.csv or cleaned_ecommerce_data.csv")
    print("in Excel or any other application, then run the program again.")
    exit()


# ---------------------------------------
# STEP 14: SAVE MODEL
# ---------------------------------------

joblib.dump(
    kmeans,
    "models/kmeans_model.pkl"
)

joblib.dump(
    scaler,
    "models/scaler.pkl"
)

print("Model saved successfully!")


# ---------------------------------------
# STEP 15: CREATE VISUALIZATIONS
# ---------------------------------------

# Customer Segment Count

plt.figure(figsize=(8, 5))

sns.countplot(
    data=final_data,
    x="Customer_Segment",
    order=[
        "Low Value Customers",
        "Occasional Customers",
        "Regular Customers",
        "VIP Customers"
    ]
)

plt.xticks(rotation=20)

plt.title("Customer Segment Distribution")

plt.xlabel("Customer Segment")
plt.ylabel("Number of Customers")

plt.tight_layout()

plt.savefig(
    "output/customer_segments.png"
)

plt.close()


# Frequency vs Monetary

plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=final_data,
    x="Frequency",
    y="Monetary",
    hue="Customer_Segment",
    s=70
)

plt.title("Customer Purchase Behavior")

plt.xlabel("Purchase Frequency")
plt.ylabel("Total Monetary Value")

plt.tight_layout()

plt.savefig(
    "output/customer_behavior.png"
)

plt.close()


# ---------------------------------------
# PROJECT COMPLETION
# ---------------------------------------

print("\nVisualizations saved successfully!")

print("\nPROJECT ANALYSIS COMPLETED SUCCESSFULLY!")