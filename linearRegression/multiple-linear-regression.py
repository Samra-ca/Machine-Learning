import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.datasets import fetch_california_housing, load_diabetes
from sklearn.preprocessing import StandardScaler

# ─────────────────────────────────────────
# DATASET 1: Diabetes Dataset (small, clean)
# ─────────────────────────────────────────
diabetes = load_diabetes()
X, y = diabetes.data, diabetes.target

print("=== Diabetes Dataset ===")
print(f"Features: {diabetes.feature_names}")
print(f"Samples:  {X.shape[0]}, Features: {X.shape[1]}\n")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(f"R²:   {r2_score(y_test, y_pred):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")

# Plot: Actual vs Predicted
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred, alpha=0.6, color='steelblue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Diabetes — Actual vs Predicted")

# Plot: Residuals
residuals = y_test - y_pred
plt.subplot(1, 2, 2)
plt.scatter(y_pred, residuals, alpha=0.6, color='orange')
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Predicted")
plt.ylabel("Residuals")
plt.title("Diabetes — Residual Plot")

plt.tight_layout()
plt.show()

# ─────────────────────────────────────────
# DATASET 2: California Housing (large, real-world)
# ─────────────────────────────────────────
housing = fetch_california_housing()
X2, y2 = housing.data, housing.target

print("\n=== California Housing Dataset ===")
print(f"Features: {housing.feature_names}")
print(f"Samples:  {X2.shape[0]}, Features: {X2.shape[1]}\n")

X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=42)

# Scale features (important for housing data)
scaler = StandardScaler()
X2_train = scaler.fit_transform(X2_train)
X2_test  = scaler.transform(X2_test)

model2 = LinearRegression()
model2.fit(X2_train, y2_train)
y2_pred = model2.predict(X2_test)

print(f"R²:   {r2_score(y2_test, y2_pred):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y2_test, y2_pred)):.4f}")

# Feature importance (coefficients)
print("\nFeature Coefficients:")
for name, coef in zip(housing.feature_names, model2.coef_):
    print(f"  {name:15s}: {coef:.4f}")

# Plot
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.scatter(y2_test, y2_pred, alpha=0.3, color='steelblue')
plt.plot([y2_test.min(), y2_test.max()], [y2_test.min(), y2_test.max()], 'r--')
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("California Housing — Actual vs Predicted")

plt.subplot(1, 2, 2)
coef_series = dict(zip(housing.feature_names, model2.coef_))
plt.barh(list(coef_series.keys()), list(coef_series.values()), color='teal')
plt.axvline(0, color='red', linestyle='--')
plt.xlabel("Coefficient value")
plt.title("Feature Importance")

plt.tight_layout()
plt.show()