
# Project Notebook Overview

This repository contains a Jupyter notebook which demonstrates the implementation of a machine learning project. Below is a summary of the content in the notebook.

## Notebook Summary

The notebook contains the following types of content:

**Markdown Cell**: ### Energy Generation: A Global perspective...
**Markdown Cell**: ### Overview

This project uses data from the World Data Institute to predict Energy Generation classes (very low, low, mid, high) for 35000 power plants located around the world. This dataset utilize...
**Markdown Cell**: ## Preprocessing Steps...
**Markdown Cell**: ### Uploading and Displaying Data...
**Code Cell**: import pandas as pd

# Load the dataset
file_path = r'C:\Users\silve\Downloads\global_power_plant_database_v_1_3\global_power_plant_database.csv'
df = pd.read_csv(file_path)

print(df.head())
print(df...
**Markdown Cell**: # Checking for missing values...
**Code Cell**: # Check for missing values
missing_df = df.isnull().sum()

# Filter only columns with missing data
missing_columns = missing_df[missing_df > 0]
print("Columns with missing values:\n", missing_columns)...
**Markdown Cell**: # Converting commissing year to plant age (age of powerplant)...
**Code Cell**: from datetime import datetime

# Get the current year
current_year = datetime.now().year

# Convert commissioning_year to plant_age by subtracting it from the current year
df['plant_age'] = current_ye...
**Code Cell**: df.info...
**Code Cell**: df.shape...
**Code Cell**: df.describe...
**Markdown Cell**: # Exploring descriptive statistics ...
**Code Cell**: # Descriptive statistics for numerical features
df.describe()

# Descriptive statistics for categorical features (like primary_fuel)
df['primary_fuel'].value_counts()

# You can also include both cate...
**Markdown Cell**: # Removing plants with multiple fuel types because distribution is not known, focsuing on the majority...
**Code Cell**: # Drop rows where either 'other_fuel1' or 'other_fuel2' is not null
df_cleaned = df[(df['other_fuel1'].isnull()) & (df['other_fuel2'].isnull())]

# Check the shape of the cleaned DataFrame
df_cleaned....
**Markdown Cell**: Grouping renewable fuels ...
**Code Cell**: # List of fuel types to group into 'Other Renewable'
renewable_fuels = ['Biomass', 'Waste', 'Geothermal', 'Cogeneration', 'Other', 'Petcoke', 'Storage', 'Wave and Tidal']

# Use .loc to avoid the Sett...
**Code Cell**: # Calculate the percentage of each primary fuel type
fuel_type_percentage = df_cleaned['primary_fuel'].value_counts(normalize=True) * 100

# Display the percentage of each fuel type
print(fuel_type_pe...
**Markdown Cell**: Calculating Average Generation from actual, using estimated if not present...
**Code Cell**: # List of actual generation columns (replace these with actual column names if available)
actual_generation_columns = [
    'generation_gwh_2013', 
    'generation_gwh_2014', 
    'generation_gwh_2015...
**Markdown Cell**: Inputing missing plant ages by mean...
**Code Cell**: # Calculate the mean of the 'plant_age' column, ignoring NaN values
mean_plant_age = df_cleaned['plant_age'].mean()

# Use .loc to safely impute missing values
df_cleaned.loc[:, 'plant_age'] = df_clea...
**Markdown Cell**: # Recalculating average generation and binning into generation classes...
**Code Cell**: # Define the columns for actual and estimated generation for the years 2013 to 2017
actual_generation_columns = [
    'generation_gwh_2013', 'generation_gwh_2014', 'generation_gwh_2015', 'generation_g...
**Code Cell**: # Select feature columns and the target column
columns_to_check = ['latitude', 'longitude', 'plant_age', 'capacity_mw', 'average_generation', 'generation_class_4']

# Check for missing values in the s...
**Markdown Cell**: # Inputting missing generation classes by mean ...
**Code Cell**: # Calculate the mean of the non-missing values in 'average_generation'
mean_average_generation = df_cleaned['average_generation'].mean()

# Fill missing values with the mean
df_cleaned['average_genera...
**Code Cell**: # Select feature columns and the target column
columns_to_check = ['latitude', 'longitude', 'plant_age', 'capacity_mw', 'average_generation', 'generation_class_4']

# Check for missing values in the s...
**Code Cell**: # Check for missing values in 'primary_fuel'
missing_fuel = df_cleaned['primary_fuel'].isnull().sum()
print(f"Missing values in 'primary_fuel': {missing_fuel}")
...
**Code Cell**: df_cleaned.shape...
**Markdown Cell**: ## Exploratory Data Analysis 
 Total Capacity by Primary Fuel Type ...
**Code Cell**: import matplotlib.pyplot as plt

# Group the data by primary fuel and calculate the total capacity for each fuel type
capacity_by_fuel_type = df_cleaned.groupby('primary_fuel')['capacity_mw'].sum().so...
**Markdown Cell**: Total Generation By Fuel Type ...
**Code Cell**: import matplotlib.pyplot as plt

# Group the data by primary fuel and calculate the total generation for each fuel type
generation_by_fuel_type = df_cleaned.groupby('primary_fuel')['average_generation...
**Markdown Cell**: Capacity vs Plant Age...
**Code Cell**: import matplotlib.pyplot as plt
import seaborn as sns  # Ensure seaborn is imported

# Filter out rows where plant_age is not missing
df_with_age = df_cleaned.dropna(subset=['plant_age'])

# Group the...
**Markdown Cell**: Generation vs Capacity by Fuel Type...
**Code Cell**: import matplotlib.pyplot as plt
import seaborn as sns

# Set the style for seaborn
sns.set(style="whitegrid")

# Create a scatter plot to visualize the relationship between generation and capacity
plt...
**Markdown Cell**: Geographic Distribution of Power Plants by Fuel type ...
**Code Cell**: # Define the categories for grouping fuel types
non_renewables = ['Oil', 'Gas', 'Coal']
df_cleaned['primary_fuel_grouped'] = df_cleaned['primary_fuel'].apply(
    lambda x: x if x in non_renewables el...
**Markdown Cell**: Couunts of Fuel Type by Class...
**Code Cell**: # Group by 'primary_fuel_grouped' and 'generation_class_4' and count occurrences
classification_counts = df_cleaned.groupby(['primary_fuel_grouped', 'generation_class_4']).size().unstack(fill_value=0)...
**Markdown Cell**: Generation By Fuel Type Classes...
**Code Cell**: import pandas as pd
import matplotlib.pyplot as plt

# Group by 'primary_fuel_grouped' and 'generation_class_4' and calculate the mean average_generation
grouped_data = df_cleaned.groupby(['primary_fu...
**Markdown Cell**: Distribution of Generation Classes...
**Code Cell**: import matplotlib.pyplot as plt
import seaborn as sns

# Count the occurrences of each class
class_counts = df_cleaned['generation_class_4'].value_counts()

# Plotting the distribution of the classes
...
**Markdown Cell**: ## Predictive Model 
# Splitting the Data...
**Code Cell**: import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle
import os

# Step 1: Encode 'primary_fuel' if not ...
**Markdown Cell**: # Running Baseline Models ...
**Code Cell**: import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn...
**Markdown Cell**: # Tuning Best Model (Random Forest) With RandomizedSearchCV...
**Code Cell**: from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

# Define the updated parameter grid for Randomized Search
param_dist = {
    'n_estimators':...
**Markdown Cell**: # Tuning Random Forest with GridSearchCV...
**Code Cell**: from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score

# Define the parameter grid for ...
**Markdown Cell**: # Tuning Random Forest with RandomziedSearchCV and data Agumentation
Augmenting the Data by tripling it with noise based on features...
**Code Cell**: import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import classifi...
**Markdown Cell**: # Retuning Model with more Data Augmentation
Augmenting by 5 folds...
**Code Cell**: 
# Function to augment data by applying random perturbations
def augment_data(df, n_augmentations=5, jitter=0.01):
    augmented_data = []
    
    for i in range(n_augmentations):
        df_aug = df...
**Markdown Cell**: # Displaying Confusion Matrix for best tuned model...
**Code Cell**: import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Make predictions on the test set
y_pred = grid_search.best_estimator_.predict(X_test_scaled)

# Co...
**Markdown Cell**: # Displaying Feature Importance For Best model...
**Code Cell**: import numpy as np
import matplotlib.pyplot as plt

# Extract the feature importances from the best model
best_rf_model = grid_search.best_estimator_
feature_importances = best_rf_model.feature_import...
**Markdown Cell**: ## Conclusions, next steps...
**Markdown Cell**: 1. Effectiveness of the Random Forest Classifier: 94.6 - Highly Effective2. 
Significance of Feature Importance: Capacity and Primary fuel 3. 
The Value of Data Augmentation and Scalability: more ro...
**Code Cell**: ...

## How to Use This Notebook

1. Install necessary packages:
   ```
   pip install -r requirements.txt
   ```
2. Open the notebook with Jupyter:
   ```
   jupyter notebook notebook_name.ipynb
   ```

## License

This project is licensed under the MIT License.
