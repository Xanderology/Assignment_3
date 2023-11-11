#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 11:32:25 2023

@author: xanderology
"""
#A key property in the mathematics of sequences and in computer algorithms is the idea of convergence. A definition of convergence is that a series converges to an asymptote such that the difference between the series and the asymptote can be made arbitrarily small with high and known probability. (30 points)


#a. In the second assignment you messed with Fibonacci numbers. Write a function that takes an integer X as an input and will return a numpy array that contains the first X numbers of the Fibonacci numbers.

import numpy as np

def generate_fibonacci(X):
    fibonacci = np.zeros(X, dtype=int)
    fibonacci[0] = 0
    if X > 1:
        fibonacci[1] = 1
        for i in range(2, X):
            fibonacci[i] = fibonacci[i-1] + fibonacci[i-2]
    return fibonacci

# Test the function with X=10
fibonacci_10 = generate_fibonacci(10)
print("First 10 Fibonacci numbers:", fibonacci_10)


#b. Use the previous function to create a numpy array with the first 20 values of the Fibonacci numbers.
# Test the function for X = 20
fibonacci_20 = generate_fibonacci(20)
print("First 20 Fibonacci numbers:", fibonacci_20)

#c. Use this numpy array to recreate the arrays from part 3 of the second assignment, one array that is the quotient of consecutive Fibonacci numbers, and the difference of the quotient of consecutive Fibonacci numbers.
quotient_fibonacci = fibonacci_20[1:] / fibonacci_20[:-1]
difference_quotient_fibonacci = np.diff(quotient_fibonacci)
print("Quotient of consecutive Fibonacci numbers:", quotient_fibonacci)
print("Difference of the quotient of consecutive Fibonacci numbers:", difference_quotient_fibonacci)

#d. Plot all 3 of these series on the same graph. You may need to adjust the parameters of the plot in order to clearly view all three series at once.

import matplotlib.pyplot as plt

# Create an array for the indices to use as x-values
indices = np.arange(1, 19)  # Adjusted to match the dimensions

# Plot the three series with appropriate x-values for quotient and difference
plt.plot(indices, fibonacci_20[:18], label='Fibonacci Numbers')
plt.plot(indices, quotient_fibonacci[:18], label='Quotient of Consecutive Fibonacci Numbers')
plt.plot(indices, difference_quotient_fibonacci, label='Difference of Quotient')

# Add labels and legend
plt.xlabel('Index')
plt.ylabel('Value')
plt.legend()

# Show the plot
plt.show()


#e. Based upon your observation, do any of these series appear to be converging? If so what values do they appear to be converging to? Feel free to reference the values of the series when determining what value it appears to be converging to.
#Observing the plot, it appears that both the quotient of consecutive Fibonacci numbers and the difference of the quotient converge to a value around 1.618, which is the golden ratio (phi). This convergence is in line with the properties of Fibonacci numbers and the golden ratio. The Fibonacci sequence converges towards the golden ratio as the index increases.



#Exploratory Data Analysis (EDA) is a critical initial step in the data analysis process. It involves systematically examining and visualizing data sets to gain insights, detect patterns, and identify anomalies. EDA helps data analysts and scientists understand the structure and characteristics of their data, making it easier to formulate hypotheses and guide subsequent analysis. (30 points)

#a. Download data about the Titanic disaster from the Kaggle study link below. You will want both the training data and the testing data. 
#https://www.kaggle.com/datasets/dbdmobile/tita111 Links to an external site.


import os
os.path.join(os.path.dirname(__file__))


# Define the file path
file_name1 = "tit_train.csv"
file_name2 = "tit_test.csv"

#b.Open both files as pandas dataframes and concatenate them together using the concatenate command from pandas. HINT: You can look up documentation about the concatenate command either on the Pandas website, or using the help() function in Python.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data into pandas DataFrames
train_data = pd.read_csv("tit_train.csv")
test_data = pd.read_csv("tit_test.csv")

# Concatenate the training and testing DataFrames
merged_df = pd.concat([train_data, test_data], ignore_index=True)

# Set a custom color palette for seaborn
sns.set_palette("Set2")

#c.Create a summary of the pandas dataframe.

summary = merged_df.describe()
print(summary)

#dCreate a histogram showing the distribution of age of people on the Titanic. Make another histogram showing the distribution of age of people on the Titanic segregated by survivalship.

# Histogram showing the distribution of age of people on the titanic
plt.figure(figsize=(10, 6))
sns.histplot(merged_df['Age'].dropna(), bins=30, kde=True, color='skyblue')
plt.xlabel('Age')
plt.ylabel('Count')
plt.title('Distribution of Age on Titanic')
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.show()

# Histogram showing the distribution of age segregated by survivalship
plt.figure(figsize=(10, 6))
sns.histplot(data=merged_df, x='Age', bins=30, hue='Survived', element='step', common_norm=False, palette=["salmon", "skyblue"])
plt.xlabel('Age')
plt.ylabel('Count')
plt.legend(title='Survived', labels=['Did Not Survive', 'Survived'])
plt.title('Distribution of Age on Titanic by Survivalship')
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.show()

# Bar chart showing the percentages of who survived
survival_percentage = merged_df['Survived'].value_counts(normalize=True) * 100
plt.figure(figsize=(8, 6))
sns.barplot(x=survival_percentage.index, y=survival_percentage.values, palette=["salmon", "skyblue"])
plt.xlabel('Survival')
plt.ylabel('Percentage')
plt.title('Percentage of Survival on Titanic')
plt.xticks([0, 1], ['Did Not Survive', 'Survived'])
plt.ylim(0, 100)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.show()


# Bar chart showing the percentages of who survived segregated by sex
survival_percentage_by_sex = merged_df.groupby('Sex')['Survived'].mean() * 100
plt.figure(figsize=(8, 6))
sns.barplot(x=survival_percentage_by_sex.index, y=survival_percentage_by_sex.values, palette="Set2")
plt.xlabel('Sex')
plt.ylabel('Percentage')
plt.title('Percentage of Survival on Titanic by Sex')
plt.ylim(0, 100)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.show()



# Boxplot showing the distribution of who survived on the Titanic vs their passenger class
plt.figure(figsize=(10, 6))
sns.boxplot(data=merged_df, x='Survived', y='Pclass', palette=["salmon", "skyblue"])
plt.xlabel('Survived')
plt.ylabel('Passenger Class')
plt.title('Distribution of Survival vs Passenger Class')
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.show()



# Boxplot showing the distribution of who survived on the Titanic vs their passenger class segregated by sex
plt.figure(figsize=(10, 6))
sns.boxplot(data=merged_df, x='Survived', y='Pclass', hue='Sex', palette="Set2")
plt.xlabel('Survived')
plt.ylabel('Passenger Class')
plt.title('Distribution of Survival vs Passenger Class by Sex')
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.show()

#Write a few sentence explaining the findings of your analysis. Feel free to reference any of visualizations.
#The histogram of age distribution shows that a significant portion of passengers were in their 20s and 30s. 
#The histogram of age distribution segregated by survivalship indicates that a higher proportion of younger passengers survived compared to older ones.
#The bar chart of survival percentages shows that a larger proportion of passengers did not survive compared to those who survived.
#The bar chart of survival percentages segregated by sex shows that a higher percentage of females survived compared to males.
#The boxplot of survival vs passenger class demonstrates that a larger proportion of passengers in the first class survived compared to the lower classes.
#The boxplot of survival vs passenger class segregated by sex further emphasizes the higher survival rate of females, particularly in the first class.

#BONUS: Additional embellishments to your visualizations, using matplotlib and/or seaborn functions to make your visualizations more aesthetically pleasing will be rewarded up to 20 extra points.
