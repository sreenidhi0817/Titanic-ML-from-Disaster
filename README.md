# Titanic-ML-from-Disaster
This project is based on Kaggle’s Titanic: Machine Learning from Disaster competition. The objective is to build a model that predicts whether a passenger survived the Titanic shipwreck, using structured data such as age, class, sex, and family relationships.

The goal was to apply classification techniques to create a predictive model for passenger survival. After experimenting with various approaches, including ensemble methods like VotingClassifier and Random Forest, the final model used was Logistic Regression. This model achieved a public leaderboard score of 0.75837 on Kaggle.

Step 1: Data Cleaning:
- Dropped columns with limited predictive value: PassengerId, Name, Ticket, Embarked, Fare, and Cabin.
- Filled missing values in the Age column using the median.
- Converted the Sex column into numeric format to ensure compatibility with machine learning models.

Step 2: Building and Testing the Model

- Selected the following features for training: Sex, Pclass, Age, SibSp, and Parch.
- Split the data into training and validation sets using an 80/20 split with train_test_split.
- Trained a Logistic Regression model using scikit-learn.
- Evaluated the model using accuracy_score on the validation set to assess performance.

Step 3: Generating Predictions

- Applied the same preprocessing steps to the test dataset.
- Used the trained model to predict survival outcomes on the test data.
- Created a submission.csv file with PassengerId and predicted Survived values for Kaggle submission.

