# Simple Linear Regression with one feature (TV vs Sales)
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Step 1: Load dataset
dataset = pd.read_csv(r"Advertising.csv")

# Step 2: Select one feature (TV) and target (Sales)
X = dataset[["TV"]].values   # Feature: TV budget
y = dataset["Sales"].values  # Target: Sales

# Step 3: Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=1/3, random_state=0
)

# Step 4: Train the Simple Linear Regression model
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Step 5: Predict the Test set results
y_pred = regressor.predict(X_test)

# Step 6: Visualize Training set results
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Sales vs TV (Training set)')
plt.xlabel('TV Advertising Budget')
plt.ylabel('Sales')
plt.show()

# Step 7: Visualize Test set results
plt.scatter(X_test, y_test, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')  # same line
plt.title('Sales vs TV (Test set)')
plt.xlabel('TV Advertising Budget')
plt.ylabel('Sales')
plt.show()

# Step 8: Print Coefficient and Intercept
print("Coefficient (slope):", regressor.coef_[0])
print("Intercept:", regressor.intercept_)

# Step 9: Single prediction
print("Predicted Sales for TV=200:", regressor.predict([[200]])[0])
