# data manipulation modules
import pandas as pd
import numpy as np

# plot modules
import matplotlib.pyplot as plt

# metrics modules
from sklearn.metrics import f1_score, accuracy_score


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



def multinomial_pdf(x_value, colour_distribution_list) -> float:
    return colour_distribution_list[x_value - 1]




def gauss_pdf(x_value, sample_mean, sample_variance) -> float:
    # calcuates the likelihood probability = p(x|omega(type)), given that x follows gaussian distribution

    term1 = 1 / (np.sqrt(2 * np.pi * (sample_variance)))
    gauss_exp_numerator = - np.power((x_value - sample_mean), 2)
    gauss_exp_denominator = 2 * (sample_variance)
    term2 = np.exp(gauss_exp_numerator / gauss_exp_denominator)


    return (term1 * term2)


def fit(df):

    calculate_type_appearances(df)
    classes = np.unique(df.type)

    samples = [np.array(df.loc[df['type'] == idx]) for idx, el in enumerate(classes)]

    ghoul_array = samples[0]
    goblin_array = samples[1]
    ghost_array = samples[2]

    colour_distribution = [len(df.loc[(df['type'] == idx) & (df['color'] == i)]) / len(df.loc[df['type'] == idx]) for idx, el in enumerate(classes) for i in np.unique(df['color'])]



    ghoul_colours =  [p for p in colour_distribution[0:6]]
    goblin_colours = [p for p in colour_distribution[6:12]]
    ghost_colours = [p for p in colour_distribution[12:]]



    colour_distribution[0] = ghoul_colours
    colour_distribution[1] = goblin_colours
    colour_distribution[2] = ghost_colours


    mean_values_ghoul = [np.mean(ghoul_array, 0)[i] for i in range(1, 5)]
    variances_ghoul = [np.var(ghoul_array, 0)[i] for i in range(1, 5)]

    mean_values_goblin = [np.mean(goblin_array, 0)[i] for i in range(1, 5)]
    variances_goblin = [np.var(goblin_array, 0)[i] for i in range(1, 5)]

    mean_values_ghost = [np.mean(ghost_array, 0)[i] for i in range(1, 5)]
    variances_ghost = [np.var(ghost_array, 0)[i] for i in range(1, 5)]




    return (mean_values_ghoul, variances_ghoul, mean_values_goblin, variances_goblin, mean_values_ghost, variances_ghost, colour_distribution)



def naive_Bayes(df, mean_values_ghoul, variances_ghoul, mean_values_goblin, variances_goblin, mean_values_ghost, variances_ghost, colour_distribution):
    global _ghoul, _goblin, _ghost
    df_to_np = np.array(df)

    for i in range(len(df)):
        ghost_probability = []
        goblin_probability = []
        ghoul_probability = []

        for j in range(1, 5):
            ghoul_probability.append(gauss_pdf(df_to_np[i][j], mean_values_ghoul[j - 1], variances_ghoul[j - 1]))
            goblin_probability.append(gauss_pdf(df_to_np[i][j], mean_values_goblin[j - 1], variances_goblin[j - 1]))
            ghost_probability.append(gauss_pdf(df_to_np[i][j], mean_values_ghost[j - 1], variances_ghost[j - 1]))

        ghoul_probability.append(multinomial_pdf(df_to_np[i][5], colour_distribution[0]))
        goblin_probability.append(multinomial_pdf(df_to_np[i][5], colour_distribution[1]))
        ghost_probability.append(multinomial_pdf(df_to_np[i][5], colour_distribution[2]))

        ghoul_ = np.array(ghoul_probability)
        goblin_ = np.array(goblin_probability)
        ghost_ = np.array(ghost_probability)


        ghoul_score = np.prod(ghoul_)
        goblin_score = np.prod(goblin_)
        ghost_score = np.prod(ghost_)

        list_score = [ghoul_score, goblin_score, ghost_score]
        idx = np.argmax(list_score)

        if idx == 0:
            _ghoul.append(df['id'][i])
        elif idx == 1:
            _goblin.append(df['id'][i])
        elif idx == 2:
            _ghost.append(df['id'][i])



def extract_predictions_csv (predicted, test_df):
    predicted_lst = [[test_df.loc[idx]['id'], el] for idx, el in enumerate(predicted)]

    for i in range(len(predicted_lst)):
        if predicted_lst[i][1] == 0:
            predicted_lst[i][1] = "Ghoul"
        if predicted_lst[i][1] == 1:
            predicted_lst[i][1] = "Goblin"
        if predicted_lst[i][1] == 2:
            predicted_lst[i][1] = "Ghost"

    # Kaggle gives 0.74480 score.
    prediction_df = pd.DataFrame(predicted_lst, columns=['id', 'type'])
    prediction_df.to_csv("output.csv", index=False)


def main(*args):
    train_df = store_data("train.csv", "train")
    naive_bayes_args = fit(train_df)

    # print(np.unique(train_df['color']))
    test_df = store_data("test.csv", "test")

    naive_Bayes(test_df, naive_bayes_args[0], naive_bayes_args[1], naive_bayes_args[2], naive_bayes_args[3]
                , naive_bayes_args[4], naive_bayes_args[5], naive_bayes_args[6])

    predicted = merge_predictions()

    extract_predictions_csv(predicted, test_df)


    #true = [train_df['type'][i] for i in range((len(train_df)))]

    # only for train
    #print("F1 score = {}".format(f1_score(true, predicted, average='weighted')))
    #print("Accuracy = {}".format(accuracy_score(true, predicted)))



if __name__ == "__main__":
	main()
