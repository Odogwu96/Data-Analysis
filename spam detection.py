import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression

email = pd.read_csv('spam_ham.csv')
print(email.shape)
print(email.columns)
email.drop("Unnamed: 0", axis=1, inplace=True)
email.drop('label', axis=1, inplace=True)
print(email.shape)
print(email.isnull().sum())
print(email.head())
print(email.dtypes)
labelencoder = LabelEncoder()
email.iloc[:, 0] = labelencoder.fit_transform(email.iloc[:, 0].values)
print(email.dtypes)
X = email.iloc[:, 1:2].values
Y =email.iloc[:, 1].values
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(5, 2))
index = 0
ax = ax.flatten()
cols = ['text', 'label_num']
for col in cols:
    minimum = min(email[col])
    maximum = max(email[col])
    email[col] = (email[col] - minimum) / (maximum - minimum)
for col, value in email.items():
    sns.countplot(y=col, data=email, ax=ax[index])
    index += 1
print(plt.tight_layout())
plt.show()
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=40)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

def models(X_train, Y_train):
    tree = DecisionTreeClassifier(criterion='entropy', random_state=0)
    tree.fit(X_train, Y_train)

    frest = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
    frest.fit(X_train, Y_train)

    lin = LinearRegression()
    lin.fit(X_train, Y_train)
    print('Decision Tree Reg Training Accuracy: ', tree.score(X_train, Y_train))
    print('Random Forest Classifier Training Accuracy: ', frest.score(X_train, Y_train))
    print('Linear Regression Accuracy: ', lin.score(X_train, Y_train))
    return tree, frest, lin
model = models(X_train, Y_train)
print(model)


print(sns)
pred = model[2].predict(X_test)
print(pred)
print()
print(Y_test)



