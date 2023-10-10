

from pandas import read_csv
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_squared_error
from joblib import dump
from sklearn.model_selection import train_test_split


df = read_csv("../datasets/pca_dataset.csv")

#create the prices category
prices_cat = [
    "0-0.1","0.1-0.2","0.2-0.3","0.3-0.4",
    "0.4-0.5","0.5-0.6","0.6-0.7","0.7-0.8",
    "0.8-0.9","0.9-1","1-1.2","1.2-1.4",
    "1.4-1.6","1.6-1.8","1.8-2","2-2.5",
    "2.5-3","3-3.5","3.5-4","4-4.5",
    "4.5-5","5-6","6-7","7-8",
    "8-9","9-10","10-15","15-20",
    "20-25","25-30","30-40","40-50",
    "50-60","60-70","70-80","80-90",
    "90-100", "100-10000000",
]

def get_category(price):
    for (i, category) in enumerate(prices_cat):
        if price <= float(category.split("-")[1]):
            return i
        
def get_cat_array(price):
    cat_array = [0 for _ in range(len(prices_cat))]
    cat_array[get_category(price)] = 1
    return cat_array

# for now, drop the columns that are not numeric
df.drop("name", axis=1, inplace=True)

# convert the dataframe to numpy arrays
prices = df["prices"].to_numpy()
y = [get_cat_array(price) for price in prices]

X = df.drop("prices", axis=1).to_numpy()

# now we can try to predict the prices
# split the data into training and testing sets
train_amount = 0.7
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_amount)

# let's try a neural network
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# now we can test the model
y_pred_train = model.predict(X_train)    # predict on the training set
tr_error = mean_squared_error(y_train, y_pred_train)    # calculate the training error
y_pred_val = model.predict(X_test) # predict values for the validation data 
val_error = mean_squared_error(y_test, y_pred_val) # calculate the validation error

print(f"Training error: {tr_error}")
print(f"Validation error: {val_error}")

# save the model
dump(model, "../models/tree_model.joblib")