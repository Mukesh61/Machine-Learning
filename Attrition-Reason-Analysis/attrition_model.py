import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

employee_df = pd.read_csv('/kaggle/input/dataset/HR-Employee-Attrition.csv')

#Check input file, like see all columns name, sample data
print(employee_df.columns)
print(employee_df['JobInvolvement'])

df = pd.get_dummies(employee_df , drop_first=True)

# get feature and target, taking attrition_yes, as output column and rest other column as feature
X = df.drop('Attrition_Yes', axis=1)
y = df['Attrition_Yes']

#divide it into training and test dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ----------------------------transform data-----------------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# ----------------------------Train the model, predict and evaluate-----------------------------------
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# ----------------------------Search for best parameter value-----------------------------------
from sklearn.model_selection import GridSearchCV

# Define the parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Perform grid search with cross-validation
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

print(grid_search.best_params_)

# ----------------------------Get important features from the trained model-----------------------------------
importances = rf_model.feature_importances_

feature_names = X.columns
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
})

# Sort by importance
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
print(feature_importance_df)

# ---------------------Plot features for visualization-------------------------------------
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(4, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
plt.title('Feature Importance in Predicting Employee Attrition')
plt.show()
