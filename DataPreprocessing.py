import pandas as pd
import numpy as np

#Original Data 
data = {
    "Age": [25, 30, 35, np.nan, 40, 28, 30],
    "Salary": [50000, 60000, 65000, 70000, np.nan, 52000, 60000],
    "Gender": ['Male', 'Female', 'Female', 'Male', 'Female', 'Male', 'Female'],
    "Purchased": ['Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'No']
}
df = pd.DataFrame(data)
print(df)

#Handling Missing Values
df['Age'] = df['Age'].fillna(df['Age'].mean(), inplace=False)
df['Salary'] = df['Salary'].fillna(df['Salary'].mean(), inplace=False)
print(df)

#Remove duplicates
df.drop_duplicates(inplace=True)
print(df)

#Encoding Categorical Variables
from sklearn.preprocessing import LabelEncoder      #pip install scikit-learn
le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])
df['Purchased'] = le.fit_transform(df['Purchased'])
print(df)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df[['Age', 'Salary']] = scaler.fit_transform(df[['Age', 'Salary']])
print(df)

#Split features and target
X = df.drop('Purchased', axis=1)
y = df['Purchased']
print(X)
print(y)

#Train Test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(X_train)
print(X_test)
print(y_train)
print(y_test)