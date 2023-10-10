

from pandas import read_csv
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from joblib import dump
from sklearn.model_selection import train_test_split

df = read_csv("../datasets/pca_dataset.csv")

# for now, drop the columns that are not numeric
df.drop("name", axis=1, inplace=True)

# convert the dataframe to numpy arrays
y = df["prices"].to_numpy()
X = df.drop("prices", axis=1).to_numpy()

# now we can try to predict the prices
# split the data into training and testing sets
train_amount = 0.7
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_amount)

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