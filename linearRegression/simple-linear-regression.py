import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Sample data
X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)
y = np.array([2.1, 4.0, 5.9, 8.1, 9.8, 12.2, 13.9, 16.1, 18.0, 19.9])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Metrics
print(f"Coefficient (slope): {model.coef_[0]:.4f}")
print(f"Intercept:           {model.intercept_:.4f}")
print(f"R² Score:            {r2_score(y_test, y_pred):.4f}")
print(f"MSE:                 {mean_squared_error(y_test, y_pred):.4f}")

# Plot
plt.figure(figsize=(8, 5))
plt.scatter(X, y, color="steelblue", label="Data points")
plt.plot(X, model.predict(X), color="red", linewidth=2, label="Regression line")
plt.xlabel("X")
plt.ylabel("y")
plt.title("Linear Regression")
plt.legend()
plt.tight_layout()
plt.show()