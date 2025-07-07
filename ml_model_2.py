# TASK: Use the titanic dataset to figure out who was more likely to survive

#----------------------------------------
# Step 1: Load and understand the data |
#----------------------------------------
import pandas as pd
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)
df.head()

# Understand the structure of the info
df.info()

# Summarise the statistics from numeric columns
df.describe()

#---------------------------------
# Step 2: Handle missing values |
#---------------------------------

# Check for missing values in each column
df.isnull().sum()

# Fill missing values in the 'Age' column with the mean age
df['Age'].fillna(df['Age'].mean(), inplace=True)

# Remove the 'Cabin' column due to too many missing values
df.drop(columns=['Cabin'], inplace=True)

#-----------------------------------
# Step 3: Analyse survival patterns|
#-----------------------------------

# Overall survival rate
df['Survived'].mean()

# Calculate survival rate by gender
df.groupby('Sex')['Survived'].mean()

# Survival rate by passenger class
df.groupby('Pclass')['Survived'].mean()

# Average age of survivors vs non-survivors
df.groupby('Survived')['Age'].mean()

# Survival rate of children (<16) vs adults (>=16)
df['IsChild'] = df['Age'] < 16
df.groupby('IsChild')['Survived'].mean()

#-----------------------------------
# Step 4: Advanced Insights (Bonus)|
#-----------------------------------

# Calculte the youngest person to survive
df[df['Survived'] == 1].sort_values(by='Age').head(1)

# Create a new column called 'AgeGroup'
def classify_age(age):
    if age < 16:
        return 'Child'
    elif age <=60: 
        return 'Adult'
    else: 
        return 'Senior'
    
df['AgeGroup'] = df['Age'].apply(classify_age)

# Analyze survival rate by AgeGroup
df.groupby('AgeGroup')['Survived'].mean()

#-----------------------------------
# Step 5: Visualisations (Optional)|
#-----------------------------------

import matplotlib.pyplot as plt
import seaborn as sns

# Plot survival rate by gender
sns.barplot(x='Sex', y='Survived', data=df)
plt.title("Survival rate by gender")
plt.show()

# Plot age distribution of survivors vs non-survivors
sns.histplot(data=df, x='Age', hue='Survived', bins=3, kde=True, element='step')
plt.title("Age Distribution: Survivors vs Non-Survivors")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()

