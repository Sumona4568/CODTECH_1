import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,classification_report
# read dataset
df = pd.read_csv('Social_Network_Ads.csv')
print(df)

# Feature separation
x = df.drop(['User ID', 'Purchased'], axis=1)
print(x)
y = df['Purchased']
print(y)

# Convert categorical to numerical
x['Gender'] = x['Gender'].replace({'Male': 0, 'Female': 1})
print(x)

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)
print(x_train)
print(x_test)
print(y_train)
print(y_test)

# Decision Tree Classifier
dt_clf = DecisionTreeClassifier(random_state=42)
dt_clf.fit(x_train, y_train)
print(x_train)
print(y_train)

# Make predictions
y_predict = dt_clf.predict(x_test)
print("Prediction",y_predict)

# Accuracy and evalute the model
accuracy = accuracy_score(y_test, y_predict)
print("Accuracy_score_decision_tree:", accuracy)
print("classification report:")
print(classification_report(y_test,y_predict))