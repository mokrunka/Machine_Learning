import sklearn
from sklearn import datasets
from sklearn import svm

cancer = datasets.load_breast_cancer()

print(cancer.feature_names)
print(cancer.target_names)

X = cancer.data
y = cancer.target

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1, random_state=42)

print(len(x_train))
print(len(x_test))
print(len(y_train))
print(len(y_test))

# for assigning a condition to the 0 or 1 of the class
condition = ['malignant', 'benign']

model = svm.SVC(C=2)

# fit the model to the training data (95% of total data)
model.fit(x_train, y_train)

# calculate the accuracy with which the model predicts the test data
accuracy = model.score(x_test, y_test)
print(accuracy)

# this is the prediction from the model (the 'answers')
predicted = model.predict(x_test)

# print out a readable summary comparing the prediction to the actual
for x in range(len(predicted)):
    print("Predicted: ", condition[predicted[x]], "Actual: ", condition[y_test[x]], "\nData: ", x_test[x])

# test