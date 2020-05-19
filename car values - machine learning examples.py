import pandas as pd
import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn import linear_model, preprocessing

car_data = pd.read_csv("car.data", sep=',')
car_data.columns = ['buying', 'maint', 'door', 'persons', 'lug_boot', 'safety', 'class']

#to use the KNN algorithm, we have to turn each column into an integer for the model to work
#convert each column from the df to a list, since fit_transform needs a list as input
le = preprocessing.LabelEncoder()
buying = le.fit_transform(list(car_data['buying']))
maint = le.fit_transform(list(car_data['maint']))
door = le.fit_transform(list(car_data['door']))
persons = le.fit_transform(list(car_data['persons']))
lug_boot = le.fit_transform(list(car_data['lug_boot']))
safety = le.fit_transform(list(car_data['safety']))
cls = le.fit_transform(list(car_data['class']))

# predict = 'class'

#create a bunch of tuple objects (zip function)
#X are the independent variables from which we want to predict the class (cls)
X = list(zip(buying, maint, door, persons, lug_boot, safety))
#y is the class we want to predict from the X data (dependent variable)
y = list(cls)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1, random_state=43)

# print(x_train)
print(x_test)
# print(y_train)
print(y_test)

model = KNeighborsClassifier(n_neighbors=9)

#fit the model to the training data (95% of total data)
model.fit(x_train, y_train)

#calculate the accuracy with which the model predicts the test data
accuracy = model.score(x_test, y_test)
print(accuracy)

#this is the prediction from the model
predicted = model.predict(x_test)
names = ['unacc', 'acc', 'good', 'vgood']

#print out a readable summary comparing the prediction to the actual
for x in range(len(predicted)):
    print("Predicted: ", names[predicted[x]], "Actual: ", names[y_test[x]], "Data: ", x_test[x])