import warnings
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
warnings.filterwarnings('ignore')

boston = pd.read_csv('Boston.csv')
boston.drop(columns=['Unnamed: 0'], axis=0, inplace=True)
print(boston.head(20))
print(boston.describe)
print(boston.info())
print(boston.isnull().sum())
fig, ax = plt.subplots(nrows=2, ncols=7, figsize=(20, 10))
index = 0
ax = ax.flatten()
cols = ['crim', 'zn', 'tax', 'black']
for col in cols:
    minimum = min(boston[col])
    maximum = max(boston[col])
    boston[col] = (boston[col] - minimum) / (maximum - minimum)
for col, value in boston.items():
    sns.boxplot(y=col, data=boston, ax=ax[index])
    index += 1
print(plt.tight_layout())
plt.show()
X = boston.iloc[:, 2:12].values
Y = boston.iloc[:, 1].values
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)
def models(X_train, Y_train):
    log = LinearRegression(random_state=0)
    log.fit(X_train, Y_train)
    return log
model = models(X_train, Y_train)
print(model)
for i in models:
    cm = confusion_matrix(Y_test, model[i].predict(X_test))
    TN, FP, FN, TP = cm.ravel()
    t_score = (TP + TN) / (TP + TN + FN + FP)
    print(cm)
    print('Model[{}] Testing Accuracy = "{}"'.format(i, t_score))


