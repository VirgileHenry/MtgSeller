

from pandas import read_csv, DataFrame
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
import numpy as np

# load the full dataset
df = read_csv("../datasets/normalized_dataset.csv")

# extract text columns, name, prices 
untouched_columns = ["name", "prices"] + [f"text_{i}" for i in range(0, 768)]

untouched = df[untouched_columns]
to_pca = df.drop(untouched_columns, axis=1)

# convert the dataframe to numpy arrays
X = to_pca.to_numpy()

N = 500
pca = PCA(n_components=N)
X_new = pca.fit_transform(X)

# plot the explained variances
fig, ax = plt.subplots(figsize=(12, 5))
color = 'tab:red'
ax.tick_params(axis='y', labelcolor=color)
ax.get_yaxis().get_major_formatter().set_useOffset(False)
ax.plot(1+np.arange(N), np.cumsum(pca.explained_variance_ratio_), color=color)
ax.set_ylabel("Cumulative explained variance ratio", color=color)
#fig.tight_layout()
plt.show()

# recombine the dataframes
new_df = DataFrame(X_new)

pca_df = untouched.join(new_df)

# save the new dataset
pca_df.to_csv("../datasets/pca_dataset.csv")

