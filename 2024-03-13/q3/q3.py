import pandas as pd
import numpy as np

df = pd.read_csv('datasheet.csv')

datasets = df.iloc[:, 0]
sensitivity_scores = df.iloc[:, 1:]

ranked_scores = sensitivity_scores.rank(axis=1, method='min')

average_rank = ranked_scores.mean()

lowest_performing = average_rank.idxmin()
highest_performing = average_rank.idxmax()

print("Average Sensitivity Rank of Each Resampling Method:")
print(average_rank)
print("\nLowest Performing Method:", lowest_performing)
print("Highest Performing Method:", highest_performing)
