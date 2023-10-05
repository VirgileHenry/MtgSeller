# File that reads in the json files and generate the csv dataset.

from pandas import read_json, DataFrame, json_normalize
from json import load as load_json

def main(path):
    """
    Read the file prices and atomic card at the given path,
    Generate the csv file at that path.
    """

    price_path = path + "oracle-cards-20231004210212.json"
    cards_path = path + "AtomicCards.json"

    target_path = path + "dataset.csv"

    with open(price_path, "r", encoding="utf-8") as prices_json:
        with open(cards_path, "r", encoding="utf-8") as atomic_json:
            
            # this data set is already nice
            prices_df = read_json(prices_json)

            #this one needs a little work
            atomic_json = load_json(atomic_json)["data"]
            atomic_df = DataFrame(atomic_json.items())
            atomic_df.columns = ["name", "data"]
            atomic_df = atomic_df["data"].explode().pipe(lambda x: json_normalize(x).set_index(x.index))

            # merge the two dataframes
            merged_df = atomic_df.merge(prices_df, on="name")

            # we still need to sort the columns out a bit (144 is way too much)
            merged_df.to_csv(target_path, index=False)



if __name__ == "__main__":
    main("../datasets/")