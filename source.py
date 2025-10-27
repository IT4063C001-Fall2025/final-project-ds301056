#!/usr/bin/env python
# coding: utf-8

# # {Fast Food & General Nutrition Analysis}📝
# 
# ![Banner](./assets/banner.jpeg)

# ## Topic
# *What problem are you (or your stakeholder) trying to address?*
# 📝 <!-- Answer Below -->

# Many people regularly consume fast food, but nutritional information is often difficult to interpret, inconsistent across sources, or overlooked when making food choices. This creates challenges for:
# 
# Consumers → making informed dietary decisions.
# 
# Healthcare professionals → guiding patients on nutrition.
# 
# Researchers/educators → understanding how fast food compares with general food supply.

# ## Project Question
# *What specific question are you seeking to answer with this project?*
# *This is not the same as the questions you ask to limit the scope of the project.*
# 📝 <!-- Answer Below -->

# What is the nutritional difference between fast food menu items and general food items, and how do these values compare against authoritative USDA standards?

# ## What would an answer look like?
# *What is your hypothesized answer to your question?*
# 📝 <!-- Answer Below -->

# the answer could look like a summary table, a bar chart showing averages, boxplots with sugar per content or a scatter plot of protein vs calories.  

# ## Data Sources
# *What 3 data sources have you identified for this project?*
# *How are you going to relate these datasets?*
# 📝 <!-- Answer Below -->

# Kaggle Dataset (CSV) — Fast Food Nutrition Dataset
# 
# https://www.kaggle.com/datasets/ulrikthygepedersen/fastfood-nutrition
# 
# Contains nutrition facts for menu items from popular fast food restaurants (calories, fat, protein, etc.)
# 
# Kaggle Dataset (CSV) — Nutritional Content of Food
# 
# https://www.kaggle.com/datasets/thedevastator/the-nutritional-content-of-food-a-comprehensive
# 
# Comprehensive dataset of nutrition facts for thousands of foods from various categories, not limited to fast food.
# 
# USDA FoodData Central API — FoodData Central
# 
# https://fdc.nal.usda.gov/

# ## Approach and Analysis
# *What is your approach to answering your project question?*
# *How will you use the identified data to answer your project question?*
# 📝 <!-- Start Discussing the project here; you can add as many code cells as you need -->

# In[9]:


# 1. Import and cleaning data
# Load the Kaggle Fast Food Nutrition dataset and General Nutrition dataset into pandas dataframes.

import pandas as pd

fast_food_df = pd.read_csv("datasets\\fastfood.csv")
general_nutrition_df = pd.read_csv("datasets\\food_nutrition.csv")

# 2. Data Cleaning
# Standardize column names and units (e.g., calories, grams of fat, sugar, protein).
# Handle missing values and normalize food names for easier comparison.

# 3. USDA API Integration
# Use the USDA FoodData Central API to fetch official nutrition facts
# for selected overlapping items (e.g., "cheeseburger", "chicken sandwich", "salad").
# Align these values with Kaggle datasets to check consistency and accuracy.

# 4. Data Merging
# Create a combined dataset where each food item has:
# - Source (Fast Food, General Food, USDA)
# - Calories, fat, sugar, protein, sodium
# Use food names as keys for merging (after text cleaning/matching).

# 5. Exploratory Data Analysis (EDA)
# - Summary statistics (mean, median, std deviation) for each nutrient by category
# - Comparisons: Fast food vs general food vs USDA benchmarks
# - Visualization: bar charts, boxplots, scatterplots, heatmaps

# 6. Answering the Question
# Quantitatively: Compare calories/fat/sugar across datasets
# Qualitatively: Show how USDA data validates or contradicts Kaggle datasets


# In[4]:


### 1. Import Libraries and Load Data


# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from scipy import stats

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set display options for better visibility
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

print("✅ Libraries imported successfully!")


# In[5]:


# Load the datasets
fast_food_df = pd.read_csv("datasets\\fastfood.csv")
general_nutrition_df = pd.read_csv("datasets\\food_nutrition.csv")

print("✅ Datasets loaded successfully!")
print(f"Fast Food Dataset: {fast_food_df.shape[0]} rows × {fast_food_df.shape[1]} columns")
print(f"General Nutrition Dataset: {general_nutrition_df.shape[0]} rows × {general_nutrition_df.shape[1]} columns")


# In[7]:


# Explore Fast Food Dataset structure
print("Fast Food Dataset Overview:")
print("="*50)
print("\nFirst 5 rows:")
display(fast_food_df.head())
print("\nData types:")
print(fast_food_df.dtypes)
print("\nBasic statistics:")
display(fast_food_df.describe().round(2))


# In[9]:


# Explore General Nutrition Dataset structure
print("General Nutrition Dataset Overview:")
print("="*50)
print("\nFirst 5 rows:")
display(general_nutrition_df.head())
print("\nColumn names:")
print(list(general_nutrition_df.columns)[:20])  # Show first 20 columns
print(f"\n... and {len(general_nutrition_df.columns)-20} more columns")
print(f"\nTotal shape: {general_nutrition_df.shape[0]} rows × {general_nutrition_df.shape[1]} columns")


# In[18]:


# Analyze missing values in Fast Food dataset
print("missing values analysis for Fast Food Dataset:")
print("="*50)

ff_missing = fast_food_df.isnull().sum()
ff_missing_pct = (ff_missing / len(fast_food_df)) * 100

# Display columns with missing values
missing_cols = ff_missing_pct[ff_missing_pct > 0].sort_values(ascending=False)
if len(missing_cols) > 0:

    # Print columns with missing values
    print("\nColumns with missing values:")
    
    for col, pct in missing_cols.items():
        print(f"  {col}: {ff_missing[col]} missing ({pct:.1f}%)")
    
    # Visualize missing values
    plt.figure(figsize=(10, 6))
    missing_cols.plot(kind='barh', color='coral')
    plt.xlabel('Percentage Missing')
    plt.title('Fast Food Dataset - Missing Values by Column')
    plt.tight_layout()
    plt.show()

else:
    print("No missing values found!")



# In[20]:


# Analyze missing values in General Nutrition dataset
print("missing values analysis for General Nutrition Dataset:")
print("="*50)

gn_missing = general_nutrition_df.isnull().sum() # sum of null values per column
gn_missing_pct = (gn_missing / len(general_nutrition_df)) * 100

# Display columns with >30% missing values
high_missing = gn_missing_pct[gn_missing_pct > 30].sort_values(ascending=False)

print(f"\nColumns with >30% missing values: {len(high_missing)}")

if len(high_missing) > 0:
    print("\nTop 10 columns with highest missing %:")
    
    # Print top 10 columns with highest missing percentage
    for col, pct in high_missing.head(10).items():
        print(f"  {col}: {pct:.1f}% missing")



# In[ ]:


# Handle missing values and duplicates -DATA CLEANING
print(" Data Cleaning ")
print("="*50)

# Handle missing fiber values by imputing with restaurant median


if 'fiber' in fast_food_df.columns:
    # Count how many 'fiber' values are missing before imputation
    fiber_missing_before = fast_food_df['fiber'].isnull().sum()

    # fill in any missing values with the median of each restaurant group
    fast_food_df['fiber'] = fast_food_df.groupby('restaurant')['fiber'].transform(
        lambda x: x.fillna(x.median())
    )

     # Count missing values again after imputation
    fiber_missing_after = fast_food_df['fiber'].isnull().sum()

     # Print how many values were successfully imputed
    print(f"✓ Imputed {fiber_missing_before - fiber_missing_after} fiber values using restaurant medians")

# Remove any duplicates
ff_duplicates_before = fast_food_df.duplicated().sum()
gn_duplicates_before = general_nutrition_df.duplicated().sum()

fast_food_df = fast_food_df.drop_duplicates()
general_nutrition_df = general_nutrition_df.drop_duplicates()

# Print how many duplicates were removed
print(f"✓ Removed {ff_duplicates_before} duplicates from Fast Food dataset")
print(f"✓ Removed {gn_duplicates_before} duplicates from General Nutrition dataset")

print("\n✅ Data cleaning completed!")


# In[27]:


# Generate statistical summaries
print("Statistical Summaries")
print("="*50)

# Fast Food Nutritional Statistics 
print("\n Fast Food - Key Nutritional Statistics:")
nutrition_cols = ['calories', 'total_fat', 'protein', 'sodium', 'sugar'] # specify relevant columns (Calories, Total Fat, Protein, Sodium, Sugar)
available_cols = [col for col in nutrition_cols if col in fast_food_df.columns]
display(fast_food_df[available_cols].describe().round(2)) # round to 2 decimal places

# General Nutrition Nutritional Statistics
print("\n General Food - Key Nutritional Statistics (per 100g):")
gn_nutrition_cols = ['Energ_Kcal', 'Protein_(g)', 'Lipid_Tot_(g)', 'Carbohydrt_(g)', 'Sodium_(mg)']
gn_available_cols = [col for col in gn_nutrition_cols if col in general_nutrition_df.columns]
display(general_nutrition_df[gn_available_cols].describe().round(2))

# Comparative Insights
print("\n Comparative Insights:")
print(f"  • Fast food averages {fast_food_df['calories'].mean():.0f} calories per item") # mean calories in fast food
print(f"  • General food averages {general_nutrition_df['Energ_Kcal'].mean():.0f} calories per 100g") # mean calories in general food

# Comparative calorie density
print(f"  • Fast food is approximately {fast_food_df['calories'].mean() / general_nutrition_df['Energ_Kcal'].mean():.1f} times more calorie-dense")

# Sodium comparison
print(f"  • Sodium concern: Fast food avg = {fast_food_df['sodium'].mean():.0f}mg (>50% of daily limit)")


# In[29]:


# Analyze data distributions
print("Distribution Analysis")
print("="*50)

# Create a 2x2 grid for plots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Fast Food - Calories
axes[0,0].hist(fast_food_df['calories'].dropna(), bins=30, color='coral', alpha=0.7, edgecolor='black')  # histogram of calories
axes[0,0].axvline(fast_food_df['calories'].mean(), color='red', linestyle='--', label=f'Mean: {fast_food_df["calories"].mean():.0f}')  # show mean line
axes[0,0].axvline(fast_food_df['calories'].median(), color='blue', linestyle='--', label=f'Median: {fast_food_df["calories"].median():.0f}')  # show median line
axes[0,0].set_xlabel('Calories')
axes[0,0].set_ylabel('Frequency')
axes[0,0].set_title('Fast Food: Calorie Distribution')
axes[0,0].legend()



# General Food - Calories
axes[0,1].hist(general_nutrition_df['Energ_Kcal'].dropna(), bins=30, color='seagreen', alpha=0.7, edgecolor='black')  # histogram of calories per 100g
axes[0,1].axvline(general_nutrition_df['Energ_Kcal'].mean(), color='red', linestyle='--', label=f'Mean: {general_nutrition_df["Energ_Kcal"].mean():.0f}')  # show mean
axes[0,1].axvline(general_nutrition_df['Energ_Kcal'].median(), color='blue', linestyle='--', label=f'Median: {general_nutrition_df["Energ_Kcal"].median():.0f}')  # show median
axes[0,1].set_xlabel('Calories per 100g')
axes[0,1].set_ylabel('Frequency')
axes[0,1].set_title('General Food: Calorie Distribution (per 100g)')
axes[0,1].legend()



# Fast Food - Sodium
axes[1,0].hist(fast_food_df['sodium'].dropna(), bins=30, color='skyblue', alpha=0.7, edgecolor='black')  # sodium histogram
axes[1,0].axvline(2300, color='red', linestyle='--', linewidth=2, label='Daily Limit')  # daily sodium limit line
axes[1,0].set_xlabel('Sodium (mg)')
axes[1,0].set_ylabel('Frequency')
axes[1,0].set_title('Fast Food: Sodium Distribution')
axes[1,0].legend()



# Fast Food - Sugar
axes[1,1].hist(fast_food_df['sugar'].dropna(), bins=30, color='plum', alpha=0.7, edgecolor='black')  # sugar histogram
axes[1,1].set_xlabel('Sugar (g)')
axes[1,1].set_ylabel('Frequency')
axes[1,1].set_title('Fast Food: Sugar Distribution')



# Adjust layout and show all plots
plt.suptitle('Nutritional Distributions Comparison', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()




# Print skewness and kurtosis metrics
print("\nDistribution Metrics:")
print(f"  Fast Food Calories - Skewness: {fast_food_df['calories'].skew():.2f}")   # measure of asymmetry
print(f"  Fast Food Calories - Kurtosis: {fast_food_df['calories'].kurtosis():.2f}") # measure of peakedness
print(f"  General Food Calories - Skewness: {general_nutrition_df['Energ_Kcal'].skew():.2f}")
print(f"  General Food Calories - Kurtosis: {general_nutrition_df['Energ_Kcal'].kurtosis():.2f}")

print("\n Both datasets show right-skewed distributions with high-calorie outliers")


# In[32]:


# Perform correlation analysis
print("Correlation Analysis")
print("="*50)

# Pick numeric columns to compare
numeric_cols = ['calories', 'total_fat', 'sat_fat', 'cholesterol', 'sodium', 'total_carb', 'sugar', 'protein']

# Keep only columns that exist in the dataset
available_numeric = [col for col in numeric_cols if col in fast_food_df.columns]

# Compute correlation matrix for numeric columns
corr_matrix = fast_food_df[available_numeric].corr()

# Plot heatmap of correlations
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})  # colored heatmap of correlations
plt.title('Fast Food Nutrients: Correlation Matrix', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

# Find strongest correlation pairs
print("\nTop Correlations (excluding self-correlations):")

corr_pairs = []  # store (col1, col2, corr_value)

# Loop through matrix to collect unique correlation pairs
for i in range(len(corr_matrix.columns)):
    for j in range(i + 1, len(corr_matrix.columns)):
        corr_pairs.append((
            corr_matrix.columns[i],
            corr_matrix.columns[j],
            corr_matrix.iloc[i, j]
        ))

# Sort pairs by absolute correlation value
corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)

# Print top 5 strongest correlations
for i, (col1, col2, corr) in enumerate(corr_pairs[:5], 1):
    print(f"  {i}. {col1} ↔ {col2}: {corr:.3f}")


# In[33]:


# Detect and analyze outliers
print("Outlier Analysis")
print("="*50)

# Function to find outliers using IQR method
def detect_outliers_iqr(df, column):
    """Detect outliers using the Interquartile Range (IQR) method"""
    Q1 = df[column].quantile(0.25)   # 25th percentile
    Q3 = df[column].quantile(0.75)   # 75th percentile
    IQR = Q3 - Q1                    # IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR     # lower limit
    upper_bound = Q3 + 1.5 * IQR     # upper limit
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]  # rows outside range
    return outliers, lower_bound, upper_bound

# Create boxplots for main nutrients
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
nutrients_to_plot = ['calories', 'total_fat', 'sodium', 'protein', 'sugar', 'total_carb']
colors = ['coral', 'lightblue', 'lightgreen', 'plum', 'gold', 'pink']

# Loop through each nutrient to plot and detect outliers
for idx, (nutrient, ax, color) in enumerate(zip(nutrients_to_plot, axes.flat, colors)):
    if nutrient in fast_food_df.columns:
        data = fast_food_df[nutrient].dropna()  # remove missing values
        ax.boxplot(data, vert=True, patch_artist=True,   # create boxplot
                   boxprops=dict(facecolor=color, alpha=0.7),
                   medianprops=dict(color='red', linewidth=2))
        ax.set_ylabel(nutrient.replace('_', ' ').title())  # clean y-axis label
        ax.set_title(f'{nutrient.replace("_", " ").title()}')  # clean title
        ax.grid(True, alpha=0.3)  # add light grid

        # Detect and count outliers
        outliers, lower, upper = detect_outliers_iqr(fast_food_df, nutrient)
        ax.text(0.02, 0.98, f'Outliers: {len(outliers)}', transform=ax.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Show all plots
plt.suptitle('Outlier Detection: Fast Food Nutrients', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

# Display top high-calorie outliers
cal_outliers, _, _ = detect_outliers_iqr(fast_food_df, 'calories')
if len(cal_outliers) > 0:
    print("\nTop 5 Highest Calorie Outliers:")
    top_outliers = cal_outliers.nlargest(5, 'calories')[['restaurant', 'item', 'calories']]
    for _, row in top_outliers.iterrows():
        print(f"  • {row['restaurant']}: {row['item'][:40]} ({row['calories']:.0f} cal)")


# In[35]:


# Compare restaurant chains
print("Restaurant Chain Comparison")
print("="*50)

# Calculate statistics by restaurant
restaurant_stats = fast_food_df.groupby('restaurant').agg({
    'calories': ['mean', 'std'],
    'sodium': 'mean',
    'total_fat': 'mean',
    'protein': 'mean'
}).round(1)

# Sort by average calories
restaurant_stats = restaurant_stats.sort_values(('calories', 'mean'), ascending=False)

print("\nAverage Nutritional Values by Restaurant (Top 10):")
display(restaurant_stats.head(10))

# Visualize restaurant comparison
fig, ax = plt.subplots(figsize=(12, 6))  # set figure size
restaurants = restaurant_stats.index[:10] # top 10 restaurants
calories_mean = restaurant_stats[('calories', 'mean')].head(10)

ax.bar(range(len(restaurants)), calories_mean, color='coral', alpha=0.7, edgecolor='black')
ax.set_xticks(range(len(restaurants))) # set x-ticks
ax.set_xticklabels(restaurants, rotation=45, ha='right') # set x-tick labels
ax.set_ylabel('Average Calories')  # set y-axis label 
ax.set_title('Average Calories by Restaurant Chain (Top 10)', fontsize=14, fontweight='bold')  # set title
ax.axhline(y=600, color='red', linestyle='--', alpha=0.5, label='600 cal reference') # reference line
ax.legend()
ax.grid(True, alpha=0.3)  # add grid
plt.tight_layout() # adjust layout
plt.show()

print(f"\n📊 Key Insights:")
print(f"  • Highest avg calories: {restaurants[0]} ({calories_mean.iloc[0]:.0f} cal)") # mean calories for top restaurant
print(f"  • All top chains average >1000mg sodium per item")


# In[38]:


# Analyze "healthy" menu items
print(" Healthier Menu Items Analysis ")
print("="*50)

# Identify salad items
salad_items = fast_food_df[fast_food_df['item'].str.contains('Salad', case=False, na=False)]
non_salad_items = fast_food_df[~fast_food_df['item'].str.contains('Salad', case=False, na=False)]

print(f"\nItems identified as salads: {len(salad_items)}")
print(f"Other items: {len(non_salad_items)}")

if len(salad_items) > 0:
    # Compare nutritional values of salads vs non-salads
    comparison_data = pd.DataFrame({
        'Salads': [
            salad_items['calories'].mean(),
            salad_items['total_fat'].mean(),
            salad_items['protein'].mean(),
            salad_items['sodium'].mean()
        ],
        'Other Items': [
            non_salad_items['calories'].mean(),
            non_salad_items['total_fat'].mean(),
            non_salad_items['protein'].mean(),
            non_salad_items['sodium'].mean()
        ]
    }, index=['Avg Calories', 'Avg Fat (g)', 'Avg Protein (g)', 'Avg Sodium (mg)'])
    
    print("\nNutritional Comparison:")
    display(comparison_data.round(1))
    
    # Statistical test
    t_stat, p_value = stats.ttest_ind(salad_items['calories'].dropna(), 
                                      non_salad_items['calories'].dropna())
    
    print(f"\nStatistical Test (Calories): p-value = {p_value:.4f}")
    if p_value < 0.05:
        print("✓ Significant difference between salads and other items")
    else:
        print("✗ No significant difference - 'Salad' doesn't mean low-calorie!")
    
    # Find high-calorie salads
    high_cal_salads = salad_items[salad_items['calories'] > 600]
    if len(high_cal_salads) > 0:
        print(f"\n High-Calorie Salads (>600 cal): {len(high_cal_salads)} items")
        for _, row in high_cal_salads.head(3).iterrows():
            print(f"  • {row['restaurant']}: {row['item'][:40]} ({row['calories']:.0f} cal)")


# In[40]:


# USDA API Integration Plan
print(" USDA API INTEGRATION STRATEGY")
print("="*50)

print("""
Implementation Plan for USDA FoodData Central API:

1. API SETUP:
   - Endpoint: https://api.nal.usda.gov/fdc/v1/
   - Register for free API key
   - Rate limit: 1000 requests/hour

2. DATA VALIDATION:
   - Query USDA for common fast food items
   - Compare nutritional values
   - Calculate accuracy metrics

3. MISSING DATA SUPPLEMENTATION:
   - Target: Fill 40% missing vitamin data
   - Priority: Vitamins A, C, D, K
   - Match by food name similarity
      
""")


# In[43]:


# My Machine Learning Plan
print(" MACHINE LEARNING STRATEGY ")
print("="*50)

# Objective:
print("\nApproach:")
print("We will use regression models to predict missing nutritional values based on existing data patterns.")
print("We will also use classification models to categorize food items into healthiness levels based on their nutritional profiles.")

# Issues to Consider:
print("\nIssues to Consider:")
print("- Correlated features can lead to multicollinearity, affecting model accuracy.")
print("- Scaling issues may occur due to different units and ranges of nutritional values.")
print("- Outliers, such as extremely high-calorie foods, can skew averages and predictions.")

# Challenges:
print("\nChallenges We May Face:")
print("- Limited or unbalanced data could make it hard for models to generalize well.")
print("- Choosing the right model and hyperparameters may require trial and error.")
print("- Ensuring clean, properly formatted data before training will be critical.")

# Summary
print("\nSummary:")
print("Our ML strategy focuses on predicting and classifying nutritional data while minimizing bias from")
print("correlated variables, scaling differences, and extreme outliers, while addressing data and model challenges.")


# 

# ## Resources and References
# *What resources and references have you used for this project?*
# 📝 <!-- Answer Below -->

# # Resources and References
# 
# ## Datasets
# - [Fast Food Nutrition Dataset (Kaggle)](https://www.kaggle.com/datasets/ulrikthygepedersen/fastfood-nutrition)  
# - [Nutritional Content of Food Dataset (Kaggle)](https://www.kaggle.com/datasets/thedevastator/the-nutritional-content-of-food-a-comprehensive)  
# - [USDA FoodData Central API](https://fdc.nal.usda.gov/api-guide.html)  
# 
# ## Tools & Libraries
# - Python 3.12  
# - pandas, matplotlib, seaborn, requests, jupyter  
# - Visual Studio Code / Jupyter Notebook  
# 
# ## Support for Editing & Clarity
# - ChatGPT (for refining project structure, clarifying wording, and drafting sections)  
# 
# ---
# 
#  **Note for Submission**  
# Make sure you run this command at the end of your notebook to generate a `.py` file for submission:  
# 
# ```bash
# !jupyter nbconvert --to python source.ipynb
# 

# In[13]:


get_ipython().system('jupyter nbconvert --to python source.ipynb')


# In[ ]:




