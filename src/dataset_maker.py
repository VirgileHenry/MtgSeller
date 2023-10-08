# File that reads in the json files and generate the csv dataset.

from pandas import read_json, DataFrame, json_normalize, to_numeric, Series
from json import load as load_json
from sentence_transformers import SentenceTransformer


# columns to keep in the final dataset
columns = [
    "colorIdentity","colors_x","convertedManaCost","firstPrinting","layout_x","manaCost","manaValue",
    "name","printings","subtypes","supertypes","text","types","power_x","toughness_x",
    "leadershipSkills.brawl","leadershipSkills.commander","leadershipSkills.oathbreaker","keywords_x",
    "legalities.commander","legalities.duel","legalities.legacy","legalities.oathbreaker","legalities.vintage",
    "legalities.explorer","legalities.gladiator","legalities.historic","legalities.historicbrawl",
    "legalities.modern","legalities.pauper","legalities.paupercommander","legalities.pioneer",
    "legalities.brawl","legalities.future","legalities.penny","legalities.standard","legalities.alchemy",
    "side","colorIndicator","loyalty_x","legalities.predh",
    "legalities.premodern","legalities.oldschool","hasAlternativeDeckLimit","defense",
    "reprint","variation","set_name","digital","rarity","artist",
    "border_color","frame","full_art","textless","booster","story_spotlight","prices",
    "frame_effects","watermark","produced_mana",
]

# columns to apply lambda functions to
lambda_columns = [
    ["manaCost", lambda x: x[1:len(x)-1].split("}{") if type(x) == str else []], # conversion between {2}{R}{R} and [2, R, R], for one hot encoding
    ["power_x", lambda x: int(x) if type(x) == str and x.isdigit() else -1],
    ["toughness_x", lambda x: int(x) if type(x) == str and x.isdigit() else -1],
    ["loyalty_x", lambda x: int(x) if type(x) == str and x.isdigit() else -1],
    ["defense", lambda x: int(x) if type(x) == str and x.isdigit() else -1],
    ["leadershipSkills.brawl", lambda x: (1 if x else 0) if type(x) == bool else 0],
    ["leadershipSkills.commander", lambda x: (1 if x else 0) if type(x) == bool else 0],
    ["leadershipSkills.oathbreaker", lambda x: (1 if x else 0) if type(x) == bool else 0],
    ["hasAlternativeDeckLimit", lambda x: (1 if x else 0) if type(x) == bool else 0],
    ["reprint", lambda x: (1 if x else 0) if type(x) == bool else 0],
    ["variation", lambda x: (1 if x else 0) if type(x) == bool else 0],
    ["digital", lambda x: (1 if x else 0) if type(x) == bool else 0],
    ["full_art", lambda x: (1 if x else 0) if type(x) == bool else 0],
    ["textless", lambda x: (1 if x else 0) if type(x) == bool else 0],
    ["booster", lambda x: (1 if x else 0) if type(x) == bool else 0],
    ["story_spotlight", lambda x: (1 if x else 0) if type(x) == bool else 0],
    ["legalities.commander", lambda x: 0 if type(x) == float else 1], # false are written as NaN float
    ["legalities.duel", lambda x: 0 if type(x) == float else 1],
    ["legalities.legacy", lambda x: 0 if type(x) == float else 1],
    ["legalities.oathbreaker", lambda x: 0 if type(x) == float else 1],
    ["legalities.vintage", lambda x: 0 if type(x) == float else 1],
    ["legalities.explorer", lambda x: 0 if type(x) == float else 1],
    ["legalities.gladiator", lambda x: 0 if type(x) == float else 1],
    ["legalities.historic", lambda x: 0 if type(x) == float else 1],
    ["legalities.historicbrawl", lambda x: 0 if type(x) == float else 1],
    ["legalities.modern", lambda x: 0 if type(x) == float else 1],
    ["legalities.pauper", lambda x: 0 if type(x) == float else 1],
    ["legalities.paupercommander", lambda x: 0 if type(x) == float else 1],
    ["legalities.pioneer", lambda x: 0 if type(x) == float else 1],
    ["legalities.brawl", lambda x: 0 if type(x) == float else 1],
    ["legalities.future", lambda x: 0 if type(x) == float else 1],
    ["legalities.penny", lambda x: 0 if type(x) == float else 1],
    ["legalities.standard", lambda x: 0 if type(x) == float else 1],
    ["legalities.alchemy", lambda x: 0 if type(x) == float else 1],
    ["legalities.predh", lambda x: 0 if type(x) == float else 1],
    ["legalities.premodern", lambda x: 0 if type(x) == float else 1],
    ["legalities.oldschool", lambda x: 0 if type(x) == float else 1],
    ["side", lambda x: x == (1 if x == "b" else 0) if type(x) == bool else 0],
    ["border_color", lambda x: x if type(x) == list else [x]],
    ["watermark", lambda x: x if type(x) == list else [x]],
    ["layout_x", lambda x: x if type(x) == list else [x]],
    ["firstPrinting", lambda x: x if type(x) == list else [x]],
    ["set_name", lambda x: x if type(x) == list else [x]],
    ["artist", lambda x: x if type(x) == list else [x]],
    ["prices", lambda x: x["eur"]]
]

# columns to one hot encode
one_hot_columns = [
    "colorIdentity","colors_x","printings","subtypes","supertypes","types","keywords_x","rarity",
    "produced_mana", "manaCost","colorIndicator","frame_effects","border_color","watermark",
    "layout_x","firstPrinting","set_name","artist"
]

def refactor_columns(df):
    for lambda_col in lambda_columns:
        (column, l) = lambda_col
        new_column = df[column].map(l)
        df = df.drop(column, axis=1)
        df = df.join(new_column)
    return df

def split_column_one_hot(df, column, drop_uniques=False):
    one_hot = df[column].str.join('|').str.get_dummies()
    # drop every column with only one card in it
    for col in one_hot.columns:
        if drop_uniques and one_hot[col].sum() < 2:
            one_hot = one_hot.drop(col, axis=1)
    # drop the original column
    df = df.drop(column, axis=1)
    # join the one hot encoded columns to the dataframe
    return df.join(one_hot, lsuffix=f"_{column}")


def one_hot_encode_df(df, drop_uniques=False):
    for column in one_hot_columns:
        df = split_column_one_hot(df, column, drop_uniques)
    return df

def main(path):
    """
    Read the file prices and atomic card at the given path,
    Generate the csv file at that path.
    """

    price_path = path + "oracle-cards-20231004210212.json"
    cards_path = path + "AtomicCards.json"

    target_path = path + "dataset.csv"

    print("Creating Dataset.")

    with open(price_path, "r", encoding="utf-8") as prices_json:
        with open(cards_path, "r", encoding="utf-8") as atomic_json:
            
            # this data set is already nice
            prices_df = read_json(prices_json)

            #this one needs a little work
            atomic_json = load_json(atomic_json)["data"]
            atomic_df = DataFrame(atomic_json.items())
            atomic_df.columns = ["name", "data"]
            atomic_df = atomic_df["data"].explode().pipe(lambda x: json_normalize(x).set_index(x.index))

            print("Json files read.")

            # merge the two dataframes
            merged_df = prices_df.merge(atomic_df, on="name")

            # we still need to sort the columns out a bit (144 is way too much)            
            final_df = merged_df[columns]
            
            print("Prices and cards merged.")

            # process text
            model = SentenceTransformer('sentence-transformers/bert-base-nli-mean-tokens')
            text_s = final_df["text"].map(lambda x: model.encode(x) if type(x) == str else model.encode(""))
            final_df.drop("text", axis=1, inplace=True)

            text_s = text_s.apply(Series).rename(columns=lambda x: f"text_{x}")
            final_df = final_df.join(text_s)

            print("Text encoded.")

            # apply lambda functions to columns
            final_df = refactor_columns(final_df)

            print("Columns refactored.")

            # one hot encode columns
            final_df = one_hot_encode_df(final_df, drop_uniques=True)

            print("Columns one hot encoded.")

            
            # clean up everything to floats (little hacky, but we are doing python, everything is hacky)
            for column in final_df.columns:
                if column != "name":
                    final_df[column] = to_numeric(final_df[column], 'coerce').fillna(-1.).astype(float)

            print("Columns cleaned.")

            # save the file
            final_df.to_csv(target_path, index=False)

            print("Dataset created.")



if __name__ == "__main__":
    main("../datasets/")

    