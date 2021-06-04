import pandas as pd
import matplotlib.pyplot as pyplot
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import sklearn
import numpy as np
from keras import metrics
from sklearn import tree

import keras
from keras.models import Sequential
from keras.layers import Dense

data = pd.read_csv("data.csv", sep=",")

predict = "result"

# Divide data into train and test splits
X = np.array(data.drop([predict], 1))
Y = np.array(data[predict])
x_train,x_test,y_train,y_test = sklearn.model_selection.train_test_split( X,Y, test_size=0.15 )




model = Sequential()
model.add(Dense(13, input_dim=2, activation='relu'))
model.add(Dense(7, kernel_initializer='normal', activation='relu'))
model.add(Dense(1, kernel_initializer='normal'))
# compile the keras model
model.compile(loss='mean_absolute_error', optimizer='adam', metrics=[metrics.mean_absolute_error])
# fit the keras model on the dataset
model.fit(x_train, y_train, validation_split=0.15, epochs=60, batch_size=10)
# evaluate the keras model
_, mse = model.evaluate(x_test, y_test)
print('Mean squared error: %.2f' % mse)

model.save("model_2")


"""
model = keras.models.load_model("my_model")
_, mse = model.evaluate(x_test, y_test)
print('Mean absolute error: %.2f' % mse)
"""




"""
fig = pyplot.figure(figsize=(10, 7))
ax = pyplot.axes(projection="3d")

# Creating plot
ax.scatter3D(data["x"], data["y"], data["result"], color="green")
pyplot.title("simple 3D scatter plot")

# show plot
pyplot.show()
"""






"""
print("Mean absolute error of cross validation score in decision tree: ", cross_val_score(DecisionTreeRegressor(), X, Y, scoring='neg_mean_absolute_error', cv=10).mean())
print("Mean absolute error of cross validation score in knn:  ", cross_val_score(KNeighborsRegressor(n_neighbors=1), X, Y, scoring='neg_mean_absolute_error', cv=10).mean())



dt = DecisionTreeRegressor()
dt.fit(X,Y)
knn = KNeighborsRegressor(n_neighbors=1)
knn.fit(X,Y)


dt_predicted = dt.predict(x_test)
knn_predicted = knn.predict(x_test)

print(sklearn.metrics.mean_absolute_error(y_test,dt_predicted))
print(sklearn.metrics.mean_absolute_error(y_test,knn_predicted))



print()
print()
print()

print(dt.predict([[17.4,23.9]])[0])
print(knn.predict([[17.4,23.9]])[0])
print(dt.predict([[52.3,98.1]])[0])
print(knn.predict([[52.3,98.1]])[0])
print(model.predict([[17.4,23.9]])[0])
print(model.predict([[52.3,98.1]])[0])
print(model.predict([[200,300]])[0])


"""
