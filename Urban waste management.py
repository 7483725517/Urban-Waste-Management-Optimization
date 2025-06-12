import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("urban waste collection.csv")
print(df.head())

df['collection_date'] = pd.to_datetime(df['collection_date'], errors='coerce', dayfirst=True)

df['fill_level_before'] = df['fill_level_before'].clip(0, 1)
df['fill_level_after'] = df['fill_level_after'].clip(0, 1)

df.sort_values(['bin_id', 'collection_date'], inplace=True)
df['prev_collection_date'] = df.groupby('bin_id')['collection_date'].shift(1)
df['days_since_last_collection'] = (df['collection_date'] - df['prev_collection_date']).dt.days

df.dropna(subset=['days_since_last_collection'], inplace=True)

waste_by_neighborhood = df.groupby('neighborhood')['waste_collected_kg'].sum().sort_values(ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x=waste_by_neighborhood.values, y=waste_by_neighborhood.index, palette='Greens', legend=False)
plt.title("Total Waste Collected by Neighborhood")
plt.xlabel("Waste Collected (kg)")
plt.ylabel("Neighborhood")
plt.tight_layout()
plt.show()

avg_days = df.groupby('bin_id')['days_since_last_collection'].mean().sort_values()

plt.figure(figsize=(12, 5))
avg_days.head(20).plot(kind='bar', color='salmon')
plt.title("Top 20 Fastest-Filling Bins (Lowest Avg. Days)")
plt.xlabel("Bin ID")
plt.ylabel("Avg. Days Between Collections")
plt.tight_layout()
plt.show()

features = ['population_density', 'collection_duration_min', 'temperature_celsius', 'humidity_percent']
X = df[features]
y = df['waste_collected_kg']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("R² Score:", r2_score(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))