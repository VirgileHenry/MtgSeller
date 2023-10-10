from pandas import read_csv, DataFrame
from sklearn import preprocessing

# load the full dataset
df = read_csv("../datasets/dataset.csv")

# extract text columns, name, prices 
untouched_columns = ["name", "prices"]

untouched = df[untouched_columns]
df = df.drop(untouched_columns, axis=1)
columns = df.columns


X = df.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
X_scaled = min_max_scaler.fit_transform(X)
df = DataFrame(X_scaled)
df.columns = columns

df = untouched.join(df)

# save the new dataset
df.to_csv("../datasets/normalized_dataset.csv")

