

from joblib import load
from pandas import read_csv

# load the model
model = load("../models/tree_model.joblib")

# load the dataset
df = read_csv("../datasets/pca_dataset.csv")

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
        price_predictions = model.predict(X)[0]

        # get the index of bigger value in price
        price = 0
        for (i, value) in enumerate(price_predictions):
            if value > price_predictions[price]:
                price = i

        # print the price
        print(f"Predicted price: {prices_cat[price]}")