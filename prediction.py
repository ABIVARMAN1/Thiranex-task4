import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, RocCurveDisplay

# 1. LOAD DATA (Assuming you've downloaded 'heart.csv' from Kaggle)
df = pd.read_csv('heart.csv')

# 2. EDA: Visualizing the impact of Age on Heart Disease
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='age', hue='target', multiple="stack", palette='magma')
plt.title('Age Distribution vs. Heart Disease Presence')
plt.show()

# 3. PREPROCESSING
X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. TRAINING THE MODEL
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 5. PREDICTION & EVALUATION
y_pred = model.predict(X_test)

print("--- HEALTHCARE MODEL REPORT ---")
print(classification_report(y_test, y_pred))

# 6. VISUALIZING PERFORMANCE: Confusion Matrix
# In Health, we must know if we are missing sick people (False Negatives)
RocCurveDisplay.from_estimator(model, X_test, y_test)
plt.title('ROC Curve for Heart Disease Prediction')
plt.show()