import pandas as pd
import matplotlib.pyplot as pyplot
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
import statistics
import sklearn
import numpy as np
from keras import metrics
import keras
from keras.models import Sequential
from keras.layers import Dense

def my_model():
    model = Sequential()
    model.add(Dense(13, input_dim=2, activation='relu'))
    model.add(Dense(8, kernel_initializer='normal', activation='relu'))
    model.add(Dense(2, kernel_initializer='normal'))
    # compile the keras model
    model.compile(loss='mean_absolute_error', optimizer='adam', metrics=[metrics.mean_absolute_error])
    return model



data = pd.read_csv("data.csv", sep=",")


fig = pyplot.figure(figsize=(10, 7))
ax = pyplot.axes(projection="3d")

# Creating plot
ax.scatter3D(data["x"], data["y"], data["result"], color="green")
pyplot.title("simple 3D scatter plot")

# show plot
pyplot.show()



predict = "result"

# Divide data into train and test splits
X = np.array(data.drop([predict], 1))
Y = np.array(data[predict])
x_train,x_test,y_train,y_test = sklearn.model_selection.train_test_split( X,Y, test_size=0.1 )


kf = KFold(n_splits=5,shuffle=True,random_state=False)
mses = []

for train_index, val_index in kf.split(x_train):
    model = my_model()
    training_x = x_train[train_index]
    training_y = y_train[train_index]
    validation_x = x_train[val_index]
    validation_y = y_train[val_index]

    model.fit(training_x, training_y, epochs=75, batch_size=10)
    _, mse = model.evaluate(validation_x, validation_y)
    mses.append(mse)
print("\n Mean absolute error of 5 fold cross validation score in deep learning model: ",statistics.mean(mses))




model = my_model()
# fit the keras model on the dataset
model.fit(x_train, y_train, epochs=75, batch_size=10)
# evaluate the keras model
_, mse = model.evaluate(x_test, y_test)
print('Mean squared error: %.f' % mse)

model.save("model_3")



model = keras.models.load_model("model_333")
_, mse = model.evaluate(x_test, y_test)
print('Mean absolute error of the deep learning model on test data: %.f' % mse)










print("Mean absolute error of 5 fold cross validation score in decision tree: ", cross_val_score(DecisionTreeRegressor(), x_train, y_train, scoring='neg_mean_absolute_error', cv=5).mean())
print("Mean absolute error of 5 fold cross validation score in knn:  ", cross_val_score(KNeighborsRegressor(n_neighbors=1), x_train, y_train, scoring='neg_mean_absolute_error', cv=5).mean())



dt = DecisionTreeRegressor()
dt.fit(x_train,y_train)
knn = KNeighborsRegressor(n_neighbors=1)
knn.fit(x_train,y_train)


dt_predicted = dt.predict(x_test)
knn_predicted = knn.predict(x_test)

print("Mean absolute error of the decision tree model on test data ",sklearn.metrics.mean_absolute_error(y_test,dt_predicted))
print("Mean absolute error of the knn model on test data",sklearn.metrics.mean_absolute_error(y_test,knn_predicted))



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



