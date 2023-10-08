

from pandas import read_csv
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from joblib import dump


df = read_csv("../datasets/pca_dataset.csv")

# for now, drop the columns that are not numeric
df.drop("name", axis=1, inplace=True)

# convert the dataframe to numpy arrays
y = df["prices"].to_numpy()
X = df.drop("prices", axis=1).to_numpy()

# new shape of X: (28402, 150)

# now we can try to predict the prices
# split the data into training and testing sets
train_amount = 0.7
train_size = int(train_amount * len(X))
print("training on {} samples".format(train_size))
X_train = X[:train_size]
X_test = X[train_size:]
y_train = y[:train_size]
y_test = y[train_size:]

"""
# we will use a simple linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# now we can test the model
score = model.score(X_test, y_test)
print(f"Score: {score}")

# Linear regression was a failure
"""

# let's try a neural network
model = MLPRegressor(hidden_layer_sizes=(1400, 1200, 1000, 800, 600, 400, 200, 100, 40, 5), max_iter=1000000)
model.fit(X_train, y_train)

# now we can test the model
y_pred_train = model.predict(X_train)    # predict on the training set
tr_error = mean_squared_error(y_train, y_pred_train)    # calculate the training error
y_pred_val = model.predict(X_test) # predict values for the validation data 
val_error = mean_squared_error(y_test, y_pred_val) # calculate the validation error

print(f"Training error: {tr_error}")
print(f"Validation error: {val_error}")

# save the model
dump(model, "../models/nn_model.joblib")