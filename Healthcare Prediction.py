from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier  # Use this when it comes to binary results (YES or NO)
# Not sure why the Classifier could work with CONTINUOUS DATA like BMI and DPF since it is use for DISCRETE DATA

health_data = pd.read_csv(r'D:\Python\Projects\Community\ADY Work\Data Science Stuff\Healthcare.csv')
x = health_data.drop(columns=['Outcome','Id'])
y = health_data['Outcome']


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=34)
model = RandomForestClassifier().fit(x_train, y_train)
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')



# Confusion matrix visualization
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# ROC Curve
y_pred_prob = model.predict_proba(x_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
roc_auc = roc_auc_score(y_test, y_pred_prob)

plt.plot(fpr, tpr, label=f'Logistic Regression (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--')  # Plotting a diagonal line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')  # If the ROC has bends or curves it means that there is a wrong prediction
plt.title('ROC Curve')
plt.legend()
plt.show()


# Showcasing the impact of each on diabetes | total sum = 100%
importances = model.feature_importances_
features = x.columns
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.title("Feature Importances")
plt.bar(range(x.shape[1]), importances[indices], align="center")
plt.xticks(range(x.shape[1]), features[indices])
plt.show()