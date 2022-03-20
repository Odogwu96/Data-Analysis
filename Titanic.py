import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import warnings
from sklearn.metrics import confusion_matrix

warnings.filterwarnings('ignore')

titanic = pd.read_csv('train.csv')
print(titanic.head(10))
print(titanic.shape)
print(titanic.describe())
print(titanic['Survived'].value_counts())
print(titanic.groupby('Sex')[['Survived']].mean())
print(titanic.pivot_table('Survived', index='Sex').plot())
sns.barplot(x='Pclass', y='Survived', data=titanic)
age = pd.cut(titanic['Age'], [0, 18, 80])
print(titanic.pivot_table('Survived', ['Sex', age], 'Pclass'))
sns.countplot(titanic['Survived'])
cols = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Embarked']
no_rows = 2
no_cols = 3
fig, axs = plt.subplots(no_rows, no_cols, figsize=(no_cols * 3.2, no_rows * 3.2))
for r in range(0, no_rows):
    for c in range(0, no_cols):
        i = r * no_cols + c
        ax = axs[r][c]
        sns.countplot(titanic[cols[i]], hue=titanic['Survived'], ax=ax)
        ax.set_title(cols[i])
        ax.legend(title='Survived', loc='upper right')
print(plt.tight_layout())
print(plt.scatter(titanic['Fare'], titanic['Pclass'], color='red', label='Passenger Paid'))
plt.ylabel('Class')
plt.xlabel('price')
plt.title('price of each class')
plt.legend()
plt.show()
print(titanic.isna().sum())
for val in titanic:
    print(titanic[val].value_counts())
    print()
titanic.dropna(subset=['Embarked', 'Cabin', 'Age'])
print(titanic.shape)
print(titanic.dtypes)
labelencoder = LabelEncoder()
titanic.iloc[:, 3] = labelencoder.fit_transform(titanic.iloc[:, 3].values)
titanic.iloc[:, 4] = labelencoder.fit_transform(titanic.iloc[:, 4].values)
titanic.iloc[:, 5] = labelencoder.fit_transform(titanic.iloc[:, 5].values)
titanic.iloc[:, 8] = labelencoder.fit_transform(titanic.iloc[:, 8].values)
titanic.iloc[:, 10] = labelencoder.fit_transform(titanic.iloc[:, 10].values)
titanic.iloc[:, 11] = labelencoder.fit_transform(titanic.iloc[:, 11].values)
print(titanic['Embarked'].unique())
print(titanic.dtypes)
X = titanic.iloc[:, 2:12].values
Y = titanic.iloc[:, 1].values
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)


def models(X_train, Y_train):
    log = LogisticRegression(random_state=0)
    log.fit(X_train, Y_train)

    knn = KNeighborsClassifier(n_neighbors=2, metric='minkowski', p=2)
    knn.fit(X_train, Y_train)

    svc_li = SVC(kernel='linear', random_state=0)
    svc_li.fit(X_train, Y_train)

    svc_rf = SVC(kernel='rbf', random_state=0)
    svc_rf.fit(X_train, Y_train)

    gauss = GaussianNB()
    gauss.fit(X_train, Y_train)

    tree = DecisionTreeClassifier(criterion='entropy', random_state=0)
    tree.fit(X_train, Y_train)

    forest = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
    forest.fit(X_train, Y_train)
    print('[0]Logistics Reg Training Accuracy: ', log.score(X_train, Y_train))
    print('[1]K Neigbours Reg Training Accuracy: ', knn.score(X_train, Y_train))
    print('[2]SVC Linear Reg Training Accuracy: ', svc_li.score(X_train, Y_train))
    print('[3]SVC RBF Reg Training Accuracy: ', svc_rf.score(X_train, Y_train))
    print('[4]Gaussian Reg Training Accuracy: ', gauss.score(X_train, Y_train))
    print('[5]Decision Tree Reg Training Accuracy: ', tree.score(X_train, Y_train))
    print('[6]Random Forest Reg Training Accuracy: ', forest.score(X_train, Y_train))

    return log, knn, svc_li, svc_rf, gauss, tree, forest


model = models(X_train, Y_train)
print(model)
for i in range(len(model)):
    cm = confusion_matrix(Y_test, model[i].predict(X_test))
    TN, FP, FN, TP = cm.ravel()
    t_scores = (TP + TN) / (TP + TN + FN + FP)
    print(cm)
    print('Model[{}] Testing Accuracy = "{}"'.format(i, t_scores))
    print()
pred = model[6].predict(X_test)
print(pred)
print()
print(Y_test)
forest = model[6]
importances = titanic.Dataframe({'feature': titanic.iloc[:, 2:12].columns, 'importance': np.round(forest.feature_importances_, 3)})
importances = importances.sort_values('importance', ascending=True).set_index('feature')
print(importances)
print(importances.plot.bar())

