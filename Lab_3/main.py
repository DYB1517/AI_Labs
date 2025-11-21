import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import mean_squared_error
from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import mean_absolute_error

df = pd.read_csv('../spaceship_titanic_changed.csv')

y = df['Transported']
X = df.drop(['Transported'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.4, random_state=42)

tree_model = DecisionTreeClassifier(max_depth= 4, max_leaf_nodes= 8)
tree_model.fit(X_test, y_test)
tree.plot_tree(tree_model)

y_proba = tree_model.predict_proba(X_test)

print('Tree_classes:', tree_model.classes_)
plt.show()

fpr, tpr, thresholds = roc_curve(y_test, y_proba[:, 1])

plt.plot(fpr, tpr, marker='o')
plt.ylim([0,1.1])
plt.xlim([0,1.1])
plt.ylabel('TPR')
plt.xlabel('FPR')
plt.title('ROC curve')

auc_metric = auc(fpr, tpr)
print('AUC: ', auc_metric)

plt.show()

y = df['FoodCourt']
X = df.drop(['FoodCourt'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.4, random_state=42)

tree_model = DecisionTreeRegressor()
tree_model.fit(X_test, y_test)
tree.plot_tree(tree_model)

y_pred_test = tree_model.predict(X_test)

MSE = mean_squared_error(y_test, y_pred_test)
RMSE = root_mean_squared_error(y_test, y_pred_test)
MAE = mean_absolute_error(y_test, y_pred_test)

print('MSE: ', MSE, ', RMSE: ', RMSE, ', MAE: ', MAE) #Mean Square Error, Root --//--, Mean Absolute Error