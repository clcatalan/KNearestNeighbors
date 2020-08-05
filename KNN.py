import sklearn  
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing


def cleanData(colName):
    #clean data via sklearn, put column values into a list, then fit_transform automatically changes it to numeric values
    le = preprocessing.LabelEncoder()
    return le.fit_transform(list(data[colName]))

data = pd.read_csv("car.data")
#print(data.head())
predict = "class"


buying = cleanData("buying")
maint = cleanData("maint")
doors = cleanData("door")
persons = cleanData("persons")
lugBoot = cleanData("lug_boot")
safety = cleanData("safety")
carClass = cleanData("class")

#zip puts all features in one list, create tuples
X = list(zip(buying, maint, doors, persons, lugBoot, safety))
y = list(carClass)

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X,y, test_size = 0.1)
#print(X_train)

model = KNeighborsClassifier(n_neighbors = 9)
model.fit(X_train, y_train)
acc = model.score(X_test, y_test)
print("Model Accuracy: ",acc)

predicted = model.predict(X_test)
names = ["unacc", "acc", "good", "vgood"]
for x in range(len(predicted)):
    print(f"[{x}]----------")
    print("Predicted: ", names[predicted[x]])
    print("Actual: ", names[y_test[x]])
    n = model.kneighbors([X_test[x]], 9, True)
    #1st array: distance to each neighbor
    #2nd array: index of that neighbor in data set
    print("N: ",n)