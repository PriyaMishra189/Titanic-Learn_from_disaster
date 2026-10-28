import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
from pandas.core.common import random_state
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score


fold=Path(__file__).parent
file=pd.read_csv(fold/'train.csv')
print(file.head())

print(file.isnull().sum())
print(file.shape)

test_file=pd.read_csv(fold/'test.csv')

print(file['Survived'].value_counts())
print(file['Survived'].value_counts(normalize=True)*100)
print(file.groupby('Sex')['Survived'].mean()*100)
print(file.groupby('Pclass')['Survived'].mean()*100)

gender_suvival=file.groupby('Sex')['Survived'].mean()*100
gender_suvival.plot(kind='bar')
plt.title('survival verfication')
plt.xlabel=('Sex')
plt.ylabel=('survival rate')
plt.box=True
plt.show()

passenger_survival=file.groupby('Pclass')['Survived'].mean()*100
passenger_survival.plot(kind='bar')
plt.title=('Class srvival')
plt.xlabel=('Passenger Class')
plt.ylabel=('Survival rate')
plt.box=True
plt.show()

file['AgeGroup']=pd.cut(file['Age'], bins=[0,10,20,40,60,100], labels=['new born', 'teenagers','Adults', 'Senior', 'Eldest'])
print(file.groupby('AgeGroup')['Survived'].mean()*100)

age_survival=file.groupby('AgeGroup')['Survived'].mean()*100
age_survival.plot(kind='bar')
plt.title=('Age wise survival')
plt.xlabel=('AgeGroups')
plt.ylabel=('Survived')
plt.show()

df=file.copy()
df['Age']=df['Age'].fillna(df['Age'].median())

df['Embarked']=df['Embarked'].fillna(df['Embarked'].mode()[0])
print(df['Embarked'])

print(df[['Age','Embarked']].isnull().sum())

df=df.drop(columns='Cabin')
print(df.isnull().sum())

df['Sex']=df['Sex'].map({'male':0,'female':1})
print(df['Sex'])

X=df[['Pclass', 'Age','Sex', 'Fare', 'SibSp', 'Parch']]
y=df['Survived']

print(X.head())
print(y.head())


X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2, random_state=42, stratify=y)
print('X_test:', X_test)
print('y_test: ', y_test)
print('X_train:', X_train)
print('y_train: ', y_train)

print("Original X:", X.shape)
print("Original y:", y.shape)
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)


model=LogisticRegression(max_iter=1000)

model.fit(X_train,y_train)

#making predictions
y_pred=model.predict(X_test)

accuracy=accuracy_score(y_test,y_pred)

print('Model Accuracy:', accuracy*100, '%')


cm=confusion_matrix(y_test,y_pred)
print(cm)

sns.heatmap(cm, annot=True, fmt="d", cmap='Blues')
plt.xlabel=("Predicted")
plt.ylabel=('Actual')
plt.title=('Titanic Survival- confusion matrix')
plt.show()

print(classification_report(y_test,y_pred))

tree_model= DecisionTreeClassifier(max_depth=4, random_state=42)

#train model
tree_model.fit(X_train, y_train)

tree_pred= tree_model.predict(X_test)

tree_accuracy= accuracy_score(y_test, y_pred)

print('Logistic regression Accuracy', accuracy*100, '%')
print('Decision tree accuracy', tree_accuracy*100, '%')

print("Logistic Regression:")
print(confusion_matrix(y_test, y_pred))

print("Decision Tree:")
print(confusion_matrix(y_test, tree_pred))

print("Identical predictions?",
      (y_pred == tree_pred).all())


print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)
print("y_pred length:", len(y_pred))
print("tree_pred length:", len(tree_pred))

print("Logistic Regression accuracy:",
      accuracy_score(y_test, y_pred))

print("Decision Tree accuracy:",
      accuracy_score(y_test, tree_pred))

print("Logistic Regression matrix:")
print(confusion_matrix(y_test, y_pred))

print("Decision Tree matrix:")
print(confusion_matrix(y_test, tree_pred))

scores= cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')

print('Scores for each fold:', scores)

print('Average accuracy', scores.mean()*100, '%')


test_features= test_file[['Pclass','Sex','Age','Fare','SibSp', 'Parch']].copy()

test_features['Sex']=test_features['Sex'].map({'male':0, 'female':1})

test_features['Fare']= test_features['Fare'].fillna(df['Fare'].median())
print(test_features['Age'].fillna(df['Age'].median()))

print(test_features.isnull().sum())
print((test_features).shape)

test_features = test_features[X.columns]

print("Training columns:", list(X.columns))
print("Test columns:", list(test_features.columns))

for col in ["Age", "Fare", "Pclass", "Sex", "SibSp", "Parch"]:
    test_features[col] = test_features[col].fillna(df[col].median())

# Match training column order
test_features = test_features[X.columns]

# Verify there are no missing values
print(test_features.isnull().sum())

# Predict
predictions = model.predict(test_features)

print(predictions[:10])
print("Total predictions:", len(predictions))



submission = pd.DataFrame({
    "PassengerId": test_file["PassengerId"],
    "Survived": predictions
})

submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Submission shape:", submission.shape)

from pathlib import Path

output_path = Path(__file__).parent / "submission.csv"
submission.to_csv(output_path, index=False)

print("Saved to:", output_path)