# https://medium.com/@guyko81/stop-predicting-numbers-start-predicting-distributions-0d4975db52ae
# https://github.com/guyko81/DistributionRegressor



"""
Predicting Distributions - pd4
DistributionRegressor: Nonparametric Distributional Regression 
Lotto 7/39 probabilistic predictions
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from distribution_regressor import DistributionRegressor

# Učitavanje stvarnih loto podataka iz CSV (bez random/sintetičkih podataka)
np.random.seed(39)
csv_path = "/data/loto7hh_4592_k27.csv"
df = pd.read_csv(csv_path)
cols = ["Num1", "Num2", "Num3", "Num4", "Num5", "Num6", "Num7"]
draws = df[cols].values.astype(float)

# Za deo vizualizacije neizvesnosti zadržavamo skalarni cilj (Num1)
X = draws[:-1]           # prethodna kombinacija
Y = draws[1:]            # sledeća kombinacija
y = Y[:, 0]              # Num1
feature_cols = [f"f{i+1}" for i in range(7)]
X_df = pd.DataFrame(X, columns=feature_cols)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_df, y, test_size=0.3, random_state=39
)

print("Training model to capture uncertainty...")
model = DistributionRegressor(
    n_estimators=200,
    learning_rate=0.1,
    verbose=0,
    random_state=39
)

model.fit(X_train, y_train)

# Make predictions
X_plot = pd.DataFrame(X_test.iloc[:100].to_numpy(), columns=feature_cols)
y_pred = model.predict(X_plot)

# Get prediction intervals
interval_50 = model.predict_interval(X_plot, confidence=0.50)  # 50% interval
lower_50, upper_50 = interval_50[:, 0], interval_50[:, 1]
interval_90 = model.predict_interval(X_plot, confidence=0.90)  # 90% interval
lower_90, upper_90 = interval_90[:, 0], interval_90[:, 1]
interval_99 = model.predict_interval(X_plot, confidence=0.99) # 99% interval
lower_99, upper_99 = interval_99[:, 0], interval_99[:, 1]

# Get quantiles
quantiles = model.predict_quantile(X_plot, q=[0.05, 0.25, 0.5, 0.75, 0.95])

print("\nPrediction Statistics:")
print(f"Mean prediction: {y_pred.mean():.3f}")
print(f"Median 90% interval width: {np.median(upper_90 - lower_90):.3f}")

# Visualize uncertainty
plt.figure(figsize=(14, 5))

# Plot 1: Prediction intervals
plt.subplot(1, 2, 1)
x_axis = np.arange(len(X_plot))
plt.scatter(np.arange(len(X_train)), y_train, alpha=0.3, s=10, label='Training data')
plt.plot(x_axis, y_pred, 'r-', linewidth=2, label='Mean prediction')
plt.fill_between(x_axis, lower_99, upper_99, alpha=0.2, color='blue', label='99% interval')
plt.fill_between(x_axis, lower_90, upper_90, alpha=0.3, color='blue', label='90% interval')
plt.fill_between(x_axis, lower_50, upper_50, alpha=0.4, color='blue', label='50% interval')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Prediction Intervals\n(Notice wider intervals where noise is higher)')
plt.legend()
plt.grid(alpha=0.3)

# Plot 2: Quantile predictions
plt.subplot(1, 2, 2)
plt.scatter(np.arange(len(X_train)), y_train, alpha=0.3, s=10, label='Training data')
plt.plot(x_axis, quantiles[:, 2], 'r-', linewidth=2, label='Median (50th)')
plt.plot(x_axis, quantiles[:, 0], 'b--', linewidth=1.5, label='5th percentile')
plt.plot(x_axis, quantiles[:, 4], 'b--', linewidth=1.5, label='95th percentile')
plt.plot(x_axis, quantiles[:, 1], 'g:', linewidth=1.5, label='25th percentile')
plt.plot(x_axis, quantiles[:, 3], 'g:', linewidth=1.5, label='75th percentile')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Quantile Predictions')
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('/uncertainty_quantification.png', dpi=150, bbox_inches='tight')
print("\n✓ Saved visualization to 'uncertainty_quantification.png'")

# Analyze uncertainty at different points
test_points = pd.DataFrame(
    [X_test.iloc[0].to_numpy(), X_test.iloc[len(X_test)//2].to_numpy(), X_test.iloc[-1].to_numpy()],
    columns=feature_cols
)
for point in test_points.index:
    p_df = pd.DataFrame([test_points.loc[point].to_numpy()], columns=feature_cols)
    interval = model.predict_interval(p_df, confidence=0.90)
    width = interval[0, 1] - interval[0, 0]
    print(f"\nAt sample idx = {point}: 90% interval width = {width:.3f}")

print("\nKey insight: The model automatically captures that predictions")
print("are more uncertain in regions with higher noise variance!")

print("\n" + "="*60)
print("Loto 7/39 - Predikcija sledeće kombinacije")
print("="*60)

# Predikcija sledeće kompletne kombinacije 7/39 (po pozicijama)
X_full = pd.DataFrame(draws[:-1], columns=feature_cols)
Y_full = draws[1:]
X_next = pd.DataFrame(draws[-1:].astype(float), columns=feature_cols)

predicted_next = []
for i in range(7):
    m = DistributionRegressor(
        n_estimators=200,
        learning_rate=0.1,
        verbose=0,
        random_state=39
    )
    m.fit(X_full, Y_full[:, i])
    # pd4: uncertainty primer -> koristimo median (q=0.5)
    p = float(m.predict_quantile(X_next, q=0.5)[0])
    predicted_next.append(int(np.clip(np.rint(p), 1, 39)))

predicted_next = np.array(predicted_next, dtype=int)
print("Predicted next loto 7/39 combination:", predicted_next)

"""
Training model to capture uncertainty...

Prediction Statistics:
Mean prediction: 5.255
Median 90% interval width: 11.020

✓ Saved visualization to 'uncertainty_quantification.png'

At sample idx = 0: 90% interval width = 13.224

At sample idx = 1: 90% interval width = 8.265

At sample idx = 2: 90% interval width = 14.327

Key insight: The model automatically captures that predictions
are more uncertain in regions with higher noise variance!

============================================================
Loto 7/39 - Predikcija sledeće kombinacije
============================================================
Predicted next loto 7/39 combination: 
[ 3  8 14 x 26 y z]
"""
