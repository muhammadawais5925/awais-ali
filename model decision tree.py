import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle

# Dataset creation
data = {
    'Category': ['Electronics', 'Clothing', 'Electronics', 'Home & Kitchen', 'Clothing', 'Electronics', 'Books', 'Home & Kitchen', 'Books', 'Electronics'],
    'Price': [1200, 50, 800, 300, 80, 1500, 20, 150, 15, 600],
    'Quantity': [1, 2, 1, 1, 3, 1, 5, 2, 10, 2],
    'Rating': [4.5, 4.0, 4.8, 3.9, 4.2, 4.7, 4.1, 4.0, 4.5, 4.3]
}
df = pd.DataFrame(data)
df['Revenue'] = df['Price'] * df['Quantity']

# Train test split
features = ['Price', 'Quantity', 'Rating', 'Revenue']
X = df[features]
y = df['Category']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Train Model
model = RandomForestClassifier(random_state=42, n_estimators=10)
model.fit(X_train, y_train)

# Predictions and validation
y_pred = model.predict(X_test)
print(f"Model Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")