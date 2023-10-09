

from joblib import load
from pandas import read_csv

# load the model
model = load("../models/nn_model.joblib")

# load the dataset
df = read_csv("../datasets/pca_dataset.csv")

while True:
    # ask the user for a card name
    card_name = input("Enter a card name: ")

    # find the card in the dataset
    card = df.loc[df["name"] == card_name]

    # if the card is not in the dataset, exit
    if card.empty:
        print("Card not found")
    else:
        # drop the name and price columns
        card = card.drop(["name", "prices"], axis=1)

        # convert the dataframe to a numpy array
        X = card.to_numpy()

        # predict the price
        price = model.predict(X)

        # print the price
        print(f"Predicted price: {price[0]}")