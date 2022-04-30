# --------- F1 SCORES (used sklearn.metrics.f1_score) ---------
# k = 1: 0.6724844583614864 
# k = 5: 0.6919439207886707 
# k = 7: 0.7088675008682425 
# k = 10: 0.7331915398994123
# k = 15: 0.7483625753068903

# --------- ACCURACY SCORES (used sklearn.metrics.accuracy_score) ---------
# k = 1: 0.6684636118598383
# k = 5: 0.6873315363881402
# k = 7: 0.706199460916442
# k = 10: 0.7304582210242587 
# k = 15: 0.7466307277628033


# data manipulation modules
import pandas as pd
import numpy as np

# plot modules
import matplotlib.pyplot as plt

# metrics modules
from sklearn.metrics import f1_score, accuracy_score

# misc modules
import csv

NUM_OF_GHOULS = 0
NUM_OF_GOBLINS = 0
NUM_OF_GHOSTS = 0

_ghoul = []
_goblin = []
_ghost = []


def merge_predictions() -> list:
    predicted = []
    # use this for metrics calculation.
    # id_type_unordered_pair = [(monster_id, monster_type) for monster_id in df['id'] for monster_type in df.loc[(df['id'] == monster_id)]["type"]]

    _ghoul_max = max(_ghoul)
    _goblin_max = max(_goblin)
    _ghost_max = max(_ghost)

    max_id = max(_ghoul_max, _goblin_max, _ghost_max)

    for i in range(0, max_id + 1):
        if i in _ghoul:
            predicted.append(0)
        if i in _goblin:
            predicted.append(1)
        if i in _ghost:
            predicted.append(2)

    return predicted



def calculate_type_appearances(df):
    global NUM_OF_GHOULS, NUM_OF_GOBLINS, NUM_OF_GHOSTS

    NUM_OF_GHOULS = len(df[df.type == 0])
    NUM_OF_GOBLINS = len(df[df.type == 1])
    NUM_OF_GHOSTS = len(df[df.type == 2])



def store_data(csv_path: str, type: str):
    df = pd.read_csv(csv_path)
    cols = df.columns
    cols = [col for col in cols]

    colour_id_list = [colour_id for colour_id in df.drop_duplicates(subset='color')['color']]

    for i in range(len(df)):
        if df.at[i, 'color'] == colour_id_list[0]:
            df.at[i, 'color'] = 1
        elif df.at[i, 'color'] == colour_id_list[1]:
            df.at[i, 'color'] = 2
        elif df.at[i, 'color'] == colour_id_list[2]:
            df.at[i, 'color'] = 3
        elif df.at[i, 'color'] == colour_id_list[3]:
            df.at[i, 'color'] = 4
        elif df.at[i, 'color'] == colour_id_list[4]:
            df.at[i, 'color'] = 5
        elif df.at[i, 'color'] == colour_id_list[5]:
            df.at[i, 'color'] = 6


        if type == "train":
            if df.at[i, 'type'] == "Ghoul":
                df.at[i, 'type'] = 0
            elif df.at[i, 'type'] == "Goblin":
                df.at[i, 'type'] = 1
            elif df.at[i, 'type'] == "Ghost":
                df.at[i, 'type'] = 2

    return df


def hamming_norm(x1, x2):
    if x1 != x2:
        return 0.05

    return 0


def euclidean_norm(x1, x2):
    return np.sqrt(np.sum(np.square(x1 - x2)))



def kNN(test_df, train_df, k: int):
    for i in range(len(test_df)):
        i_color = test_df.loc[i]['color']
        i_bone_length = test_df.loc[i]['bone_length']
        i_rotting_flesh = test_df.loc[i]['rotting_flesh']
        i_hair_length = test_df.loc[i]['hair_length']
        i_has_soul = test_df.loc[i]['has_soul']

        color_array = []
        attributes_distance_array = []

        for j in range(len(train_df)):
            j_color = train_df.loc[j]['color']
            j_bone_length = train_df.loc[j]['bone_length']
            j_rotting_flesh = train_df.loc[j]['rotting_flesh']
            j_hair_length = train_df.loc[j]['hair_length']
            j_has_soul = train_df.loc[j]['has_soul']


            has_same_color = hamming_norm(i_color, j_color)
            attributes_distance = euclidean_norm(np.array([i_bone_length, i_rotting_flesh, i_hair_length, i_has_soul]),
                                                 np.array([j_bone_length, j_rotting_flesh, j_hair_length, j_has_soul]))

            attributes_distance_array.append(has_same_color + attributes_distance)

        attributes_distance_nparray = np.array(attributes_distance_array)
        k_lowest_values_nparray = np.sort(np.argpartition(attributes_distance_nparray, k)[:k])




        decision_list = [train_df.loc[j]['type'] for j in k_lowest_values_nparray]


        idx = max(set(decision_list), key=decision_list.count)

        if idx == 0:
            _ghoul.append(test_df['id'][i])
        elif idx == 1:
            _goblin.append(test_df['id'][i])
        elif idx == 2:
            _ghost.append(test_df['id'][i])




def extract_predictions_csv (predicted, test_df):
    predicted_lst = [[test_df.loc[idx]['id'], el] for idx, el in enumerate(predicted)]

    for i in range(len(predicted_lst)):
        if predicted_lst[i][1] == 0:
            predicted_lst[i][1] = "Ghoul"
        if predicted_lst[i][1] == 1:
            predicted_lst[i][1] = "Goblin"
        if predicted_lst[i][1] == 2:
            predicted_lst[i][1] = "Ghost"

    # Kaggle gives: 0.73345 score for kNN (k = 7).
    prediction_df = pd.DataFrame(predicted_lst, columns=['id', 'type'])
    prediction_df.to_csv("output.csv", index=False)


def main(*args):

    train_df = store_data("train.csv", "train")
    test_df = store_data("test.csv", "test")

    kNN(test_df, train_df, 7)

    predicted = merge_predictions()
    extract_predictions_csv(predicted, test_df)



if __name__ == "__main__":
	main()
