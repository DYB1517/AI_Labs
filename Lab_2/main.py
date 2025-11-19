import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score

df = pd.read_csv('../spaceship_titanic_changed.csv')

print('Regression:')

y = df['Age']

X = df.drop(['Age'], axis=1)
#X = pd.get_dummies(X, drop_first=True)
#X = X.fillna(0)
#y = y.fillna(y.mean())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.4, random_state=42)

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

y_pred_test = linear_model.predict(X_test)

MSE = mean_squared_error(y_test, y_pred_test)
RMSE = root_mean_squared_error(y_test, y_pred_test)
MAE = mean_absolute_error(y_test, y_pred_test)

print('MSE: ', MSE, ', RMSE: ', RMSE, ', MAE: ', MAE)
print('Normalization:')

y_norm = df['VIP']
X_norm = df.drop(['VIP'], axis=1)

X_train_norm, X_test_norm, y_train_norm, y_test_norm = train_test_split(X_norm, y_norm, test_size=0.4, random_state=42)
X_test_norm, X_val_norm, y_test_norm, y_val_norm = train_test_split(X_test_norm, y_test_norm, test_size=0.4, random_state=42)

from sklearn.linear_model import LogisticRegression

logreg_model = LogisticRegression(max_iter = 5000)
logreg_model.fit(X_train_norm, y_train_norm)
y_pred_test_norm = logreg_model.predict(X_test_norm)

accuracy = accuracy_score(y_test_norm, y_pred_test_norm)

print('Normalization accuracy: ', accuracy)