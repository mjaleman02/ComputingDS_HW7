import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import json

df = pd.read_csv('sample_diabetes_mellitus_data.csv', delimiter = ';')  
df = df.select_dtypes(include=['number'])
X = df.drop(columns='diabetes_mellitus')  
y = df['diabetes_mellitus']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

joblib.dump(model, 'model.pkl')
with open("features.json", "w") as f:
    json.dump(X.columns.tolist(), f)

print("Model and feature names saved.")



