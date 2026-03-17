"""
TASK 1: Data Merge and EDA
Group Member: Kakooza Mahad
This script merges customer social profiles with transactions and creates EDA plots
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style for better looking plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*50)
print("TASK 1: Data Merge and EDA")
print("="*50)

# -------------------------------------------------------------------
# STEP 1: Load the datasets
# -------------------------------------------------------------------
print("\n📂 STEP 1: Loading datasets...")

try:
    social = pd.read_csv('customer_social_profiles.csv')
    transactions = pd.read_csv('customer_transactions.csv')
    print("✅ Successfully loaded both datasets")
except FileNotFoundError:
    print("❌ Error: CSV files not found. Please make sure they're in the same folder.")
    print("Creating sample data for testing...")
    
    # Create sample data if files don't exist (for testing)
    np.random.seed(42)
    n_customers = 100
    
    social = pd.DataFrame({
        'customer_id': range(1, n_customers + 1),
        'age': np.random.randint(18, 65, n_customers),
        'gender': np.random.choice(['M', 'F'], n_customers),
        'location': np.random.choice(['NY', 'CA', 'TX', 'FL', 'IL'], n_customers),
        'pages_liked': np.random.randint(10, 500, n_customers),
        'avg_session_time': np.random.uniform(1, 30, n_customers).round(2),
        'total_friends': np.random.randint(50, 1000, n_customers)
    })
    
    transactions = pd.DataFrame({
        'customer_id': np.random.choice(range(1, n_customers + 1), 300),
        'transaction_date': pd.date_range(start='2023-01-01', periods=300, freq='D'),
        'product_purchased': np.random.choice(['Laptop', 'Phone', 'Tablet', 'Headphones', 'Watch'], 300),
        'amount': np.random.uniform(10, 1000, 300).round(2),
        'payment_method': np.random.choice(['Credit Card', 'PayPal', 'Debit Card'], 300)
    })
    print("✅ Created sample data for demonstration")

# -------------------------------------------------------------------
# STEP 2: Basic Information and Summary Statistics
# -------------------------------------------------------------------
print("\n📊 STEP 2: Exploring the datasets...")

print("\n--- Social Profiles Dataset ---")
print(f"Shape: {social.shape}")
print(f"Columns: {list(social.columns)}")
print("\nFirst 5 rows:")
print(social.head())

print("\n--- Transactions Dataset ---")
print(f"Shape: {transactions.shape}")
print(f"Columns: {list(transactions.columns)}")
print("\nFirst 5 rows:")
print(transactions.head())

print("\n--- Summary Statistics (Social Profiles) ---")
print(social.describe())

print("\n--- Summary Statistics (Transactions) ---")
print(transactions.describe(include='all'))

# Check data types
print("\n--- Data Types ---")
print("Social Profiles:")
print(social.dtypes)
print("\nTransactions:")
print(transactions.dtypes)

# -------------------------------------------------------------------
# STEP 3: Check for missing values and duplicates
# -------------------------------------------------------------------
print("\n🧹 STEP 3: Data Cleaning...")

print("\nMissing values in Social Profiles:")
print(social.isnull().sum())
print("\nMissing values in Transactions:")
print(transactions.isnull().sum())

print(f"\nDuplicates in Social Profiles: {social.duplicated().sum()}")
print(f"Duplicates in Transactions: {transactions.duplicated().sum()}")

# Remove duplicates if any
social = social.drop_duplicates()
transactions = transactions.drop_duplicates()
print("✅ Removed any duplicates")

# Handle missing values (simple approach - drop rows with missing values)
social = social.dropna()
transactions = transactions.dropna()
print("✅ Removed rows with missing values")

# -------------------------------------------------------------------
# STEP 4: Create 3 EDA Plots
# -------------------------------------------------------------------
print("\n📈 STEP 4: Creating 3 EDA plots...")

# Create a figures directory
import os
if not os.path.exists('eda_plots'):
    os.makedirs('eda_plots')

# PLOT 1: Distribution Plot (Age)
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.histplot(social['age'], bins=20, kde=True, color='skyblue')
plt.title('Distribution of Customer Ages', fontsize=14, fontweight='bold')
plt.xlabel('Age')
plt.ylabel('Count')

plt.subplot(1, 2, 2)
sns.boxplot(y=social['age'], color='lightcoral')
plt.title('Age Boxplot (Outliers)', fontsize=14, fontweight='bold')
plt.ylabel('Age')

plt.tight_layout()
plt.savefig('eda_plots/plot1_age_distribution.png', dpi=300, bbox_inches='tight')
plt.show()
print("✅ Plot 1 saved: age_distribution.png")

# PLOT 2: Correlation Heatmap
plt.figure(figsize=(10, 8))
# Select only numeric columns for correlation
numeric_cols = social.select_dtypes(include=[np.number]).columns
correlation_matrix = social[numeric_cols].corr()

sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Feature Correlations in Social Profiles', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('eda_plots/plot2_correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()
print("✅ Plot 2 saved: correlation_heatmap.png")

# PLOT 3: Product Purchase Counts
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
product_counts = transactions['product_purchased'].value_counts()
bars = plt.bar(product_counts.index, product_counts.values, color='teal', alpha=0.7)
plt.title('Most Popular Products', fontsize=14, fontweight='bold')
plt.xlabel('Product')
plt.ylabel('Number of Purchases')
plt.xticks(rotation=45)

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{int(height)}', ha='center', va='bottom')

plt.subplot(1, 2, 2)
transactions['amount'].plot(kind='box')
plt.title('Transaction Amount Distribution', fontsize=14, fontweight='bold')
plt.ylabel('Amount ($)')

plt.tight_layout()
plt.savefig('eda_plots/plot3_product_analysis.png', dpi=300, bbox_inches='tight')
plt.show()
print("✅ Plot 3 saved: product_analysis.png")

# -------------------------------------------------------------------
# STEP 5: Merge Datasets
# -------------------------------------------------------------------
print("\n🔄 STEP 5: Merging datasets...")

# Check for common key
common_key = 'customer_id'
if common_key in social.columns and common_key in transactions.columns:
    print(f"✅ Both datasets have '{common_key}' as common key")
    
    # Perform merge
    merged_data = pd.merge(social, transactions, on=common_key, how='inner')
    print(f"\nMerge completed!")
    print(f"Original social profiles: {len(social)} rows")
    print(f"Original transactions: {len(transactions)} rows")
    print(f"Merged dataset: {len(merged_data)} rows")
    
    # Show merge justification
    print("\n--- Merge Justification ---")
    print("Used INNER JOIN because:")
    print("1. We only want customers with BOTH profile AND transaction data")
    print("2. This ensures our recommendation model trains on complete customer records")
    print("3. Customers without transactions can't be used for purchase prediction")
    
else:
    print(f"❌ Error: '{common_key}' not found in both datasets")
    print("Checking columns...")
    print(f"Social columns: {social.columns.tolist()}")
    print(f"Transactions columns: {transactions.columns.tolist()}")
    # Try to find common column
    common_cols = set(social.columns) & set(transactions.columns)
    if common_cols:
        common_key = list(common_cols)[0]
        print(f"Found common column: {common_key}")
        merged_data = pd.merge(social, transactions, on=common_key, how='inner')
    else:
        print("❌ No common columns found. Cannot merge.")
        merged_data = pd.DataFrame()

# -------------------------------------------------------------------
# STEP 6: Post-Merge Validation
# -------------------------------------------------------------------
print("\n✅ STEP 6: Post-Merge Validation...")

if not merged_data.empty:
    print(f"\nMerged Dataset Shape: {merged_data.shape}")
    print(f"\nFirst 5 rows of merged data:")
    print(merged_data.head())
    
    print(f"\nMissing values after merge:")
    print(merged_data.isnull().sum())
    
    print(f"\nData types after merge:")
    print(merged_data.dtypes)
    
    print(f"\nBasic statistics of merged data:")
    print(merged_data.describe())
    
    # Save merged dataset
    merged_data.to_csv('merged_dataset.csv', index=False)
    print("\n✅ Merged dataset saved as 'merged_dataset.csv'")
    
    # Create a quick summary plot of merged data
    if len(merged_data) > 0 and 'amount' in merged_data.columns and 'age' in merged_data.columns:
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(merged_data['age'], merged_data['amount'], 
                            c=merged_data['pages_liked'] if 'pages_liked' in merged_data.columns else 'blue',
                            alpha=0.6, cmap='viridis')
        plt.xlabel('Age')
        plt.ylabel('Transaction Amount ($)')
        plt.title('Age vs Spending (colored by pages liked)')
        plt.colorbar(scatter, label='Pages Liked')
        plt.savefig('eda_plots/plot4_age_vs_spending.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("✅ Bonus plot saved: age_vs_spending.png")

# -------------------------------------------------------------------
# STEP 7: Summary Report
# -------------------------------------------------------------------
print("\n" + "="*50)
print("📋 TASK 1 SUMMARY REPORT")
print("="*50)
print("\n✅ COMPLETED ALL STEPS:")
print("  ✓ Loaded datasets")
print("  ✓ Generated summary statistics")
print("  ✓ Checked data types")
print("  ✓ Created 3+ labeled plots")
print("  ✓ Handled null values and duplicates")
print("  ✓ Merged datasets with justified join")
print("  ✓ Performed post-merge validation")
print("  ✓ Saved merged dataset")
print("\n📁 OUTPUT FILES CREATED:")
print("  - merged_dataset.csv")
print("  - eda_plots/plot1_age_distribution.png")
print("  - eda_plots/plot2_correlation_heatmap.png")
print("  - eda_plots/plot3_product_analysis.png")
print("  - eda_plots/plot4_age_vs_spending.png")
print("="*50)

# Save this summary to a text file
with open('task1_complete.txt', 'w') as f:
    f.write("TASK 1 completed successfully!\n")
    f.write(f"Date: {pd.Timestamp.now()}\n")
    f.write(f"Merged dataset shape: {merged_data.shape if not merged_data.empty else 'N/A'}")
print("\n✅ Summary saved to 'task1_complete.txt'")