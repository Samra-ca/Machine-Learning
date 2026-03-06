from sklearn.linear_model import LogisticRegression
import numpy as np

# Train the model
X_train = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
y_train = np.array([0, 0, 1, 1])

model = LogisticRegression()
model.fit(X_train, y_train)

# New sample
new_sample = np.array([[2.5, 3.5]])

# METHOD 1: Using scikit-learn's built-in function
sklearn_proba = model.predict_proba(new_sample)
print(f"scikit-learn probability: {sklearn_proba[0]}")

# METHOD 2: Manually doing the linear + sigmoid steps
# Step 1: Get the coefficients and intercept from the trained model
coefficients = model.coef_[0]  # [w1, w2]
intercept = model.intercept_[0]  # b

# Step 2: Linear combination (linear regression part)
z = np.dot(new_sample, coefficients) + intercept
print(f"Linear combination (z): {z[0]:.4f}")

# Step 3: Apply sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

probability_class_1 = sigmoid(z)[0]
probability_class_0 = 1 - probability_class_1

print(f"Manual probability: [{probability_class_0:.4f}, {probability_class_1:.4f}]")

# Verify they match
print(f"\nDo they match? {np.allclose(sklearn_proba[0], [probability_class_0, probability_class_1])}")