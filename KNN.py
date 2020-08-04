import sklearn  
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing


    


data = pd.read_csv("car.data")
#print(data.head())
predict = "class"

le = preprocessing.LabelEncoder()
#clean data via sklearn, put column values into a list, then fit_transform automatically changes it to numeric values
buying = le.fit_transform(list(data["buying"]))
maint = le.fit_transform(list(data["maint"]))
doors = le.fit_transform(list(data["door"]))
persons = le.fit_transform(list(data["persons"]))
lugBoot = le.fit_transform(list(data["lug_boot"]))
safety = le.fit_transform(list(data["safety"]))
carClass = le.fit_transform(list(data["class"]))
print(buying)

#zip puts all features in one list, create tuples
X = list(zip(buying, maint, doors, persons, lugBoot, safety))
y = list(carClass)

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X,y, test_size = 0.1)
#print(X_train)

model = KNeighborsClassifier(n_neighbors = 9)
model.fit(X_train, y_train)
acc = model.score(X_test, y_test)
print(acc)

predicted = model.predict(X_test)
names = ["unacc", "acc", "good", "vgood"]
for x in range(len(predicted)):
    print("Predicted: ", names[predicted[x]], "Data: ", X_test[x], "Actual: ", names[y_test[x]])
    n = model.kneighbors([X_test[x]], 9, True)
    print("N: ",n)