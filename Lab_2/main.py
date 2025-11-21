import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score
from sklearn.linear_model import Ridge
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

df = pd.read_csv('../spaceship_titanic_changed.csv')

print('Regression:')

y = df['Age']
X = df.drop(['Age'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.4, random_state=42)

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

y_pred_test = linear_model.predict(X_test)

MSE = mean_squared_error(y_test, y_pred_test)
RMSE = root_mean_squared_error(y_test, y_pred_test)
MAE = mean_absolute_error(y_test, y_pred_test)

print('MSE: ', MSE, ', RMSE: ', RMSE, ', MAE: ', MAE) #Mean Square Error, Root --//--, Mean Absolute Error

l2_linear_model = Ridge(alpha=2.0)

l2_linear_model.fit(X_train, y_train)
y_pred_test = l2_linear_model.predict(X_test)

l2_MSE = mean_squared_error(y_test, y_pred_test)
l2_RMSE = root_mean_squared_error(y_test, y_pred_test)
l2_MAE = mean_absolute_error(y_test, y_pred_test)

print('l2_MSE: ', l2_MSE, ', l2_RMSE: ', l2_RMSE, ', l2_MAE: ', l2_MAE) #Mean Square Error, Root --//--, Mean Absolute Error

print('Clasification:')

y_clas = df['Transported']
X_clas = df.drop(['Transported'], axis=1)

X_train_clas, X_test_clas, y_train_clas, y_test_clas = train_test_split(X_clas, y_clas, test_size=0.4, random_state=42)
X_test_clas, X_val_clas, y_test_clas, y_val_clas = train_test_split(X_test_clas, y_test_clas, test_size=0.4, random_state=42)

logreg_model = LogisticRegression(max_iter = 5000)
logreg_model.fit(X_train_clas, y_train_clas)
y_pred_test_clas = logreg_model.predict(X_test_clas)

accuracy = accuracy_score(y_test_clas, y_pred_test_clas)

print('Clasification accuracy: ', accuracy)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test_clas, y_pred_test_clas)

plt.figure(figsize=(4, 3))
sns.heatmap(cm, annot=True, fmt='d', cmap='bwr')
plt.title('Confusion matrix')
plt.ylabel('True label')
plt.xlabel('Predicted label')

report = classification_report(y_test_clas, y_pred_test_clas)

print(report)

plt.show()