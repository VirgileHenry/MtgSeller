

from pandas import read_csv, DataFrame
from sklearn.decomposition import PCA

# load the full dataset
df = read_csv("../datasets/dataset.csv")

# extract text columns, name, prices 
untouched_columns = ["name", "prices"] + [f"text_{i}" for i in range(0, 768)]

untouched = df[untouched_columns]
to_pca = df.drop(untouched_columns, axis=1)

# convert the dataframe to numpy arrays
X = to_pca.to_numpy()

# all of the data is one hot encoded, let's try to reduce the dimensionality
# original shape of X: (28402, 2867)
N = 700
pca = PCA(n_components=N)
X_new = pca.fit_transform(X)

# recombine the dataframes
new_df = DataFrame(X_new)

pca_df = untouched.join(new_df)

# save the new dataset
pca_df.to_csv("../datasets/pca_dataset.csv")