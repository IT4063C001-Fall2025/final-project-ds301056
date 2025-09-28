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

# In[ ]:


# 1. Import and cleaning data
# Load the Kaggle Fast Food Nutrition dataset and General Nutrition dataset into pandas dataframes.

import pandas as pd

fast_food_df = pd.read_csv("datasets/fast_food_nutrition.csv")
general_nutrition_df = pd.read_csv("datasets/general_nutrition.csv")

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


# In[ ]:





# ## Resources and References
# *What resources and references have you used for this project?*
# 📝 <!-- Answer Below -->

# In[1]:


#Resources and References

#Datasets:

Fast Food Nutrition Dataset (Kaggle)

Nutritional Content of Food Dataset (Kaggle)

USDA FoodData Central API

#Tools & Libraries:

Python 3.12

pandas, matplotlib, seaborn, requests, jupyter

Visual Studio Code / Jupyter Notebook

#Support for Editing & Clarity:

ChatGPT (for refining project structure, clarifying wording, and drafting)

get_ipython().system('jupyter nbconvert --to python source.ipynb')


# In[ ]:




