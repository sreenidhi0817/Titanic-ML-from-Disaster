import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

pd.set_option('display.max_columns', None)

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

print("Train:")
print(train_df.head())

print("Test:")
print(test_df.head())

#data cleaning
""" find all columns that have an effect:
columns that have an effect are sex, cabin, age, pclass, sibsp, parch """

print("Data with deleted columns:")
train_df = train_df.drop(['PassengerId', 'Name', 'Ticket', 'Embarked', 'Fare', 'Cabin'], axis = 1)# delete columns that don't have an effect
print(train_df.head())

# fill with empty places for age with medians
train_df['Age'] = train_df['Age'].fillna(train_df['Age'].median())

# convert categorical values to numbers
train_df['Sex'] = train_df['Sex'].map({'male':0, 'female': 1})

# check the clean data
print('Checking clean data')
print(train_df.info())
print(train_df.head())


#features for prediction
features = ['Sex', 'Pclass', 'Age', 'SibSp', 'Parch']

# x= inputs, y = target
x = train_df[features]
y = train_df['Survived']



#import all modules
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

x_train, x_val, y_train, y_val = train_test_split(x,y,test_size=0.2, random_state=42)


#clean data before predicting from test.csv
test_df = test_df.drop(['PassengerId', 'Name', 'Ticket', 'Embarked', 'Fare', 'Cabin'], axis = 1)
test_df['Age'] = test_df['Age'].fillna(train_df['Age'].median())
test_df['Sex'] = test_df['Sex'].map({'male':0, 'female': 1})
x_test = test_df[features]


log_model = LogisticRegression()
log_model.fit(x_train,y_train)
log_preds = log_model.predict(x_val)
print("Logistic Regression Accuracy:", accuracy_score(y_val, log_preds))


log_model.fit(x_train, y_train)
predictions = log_model.predict(x_test)

test = pd.read_csv("test.csv")

submission = pd.DataFrame({
    'PassengerId': test['PassengerId'],
    'Survived': predictions
})

submission.to_csv("submission.csv", index=False)
print("submission.csv created successfully")
