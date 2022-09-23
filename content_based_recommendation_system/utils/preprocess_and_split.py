from config import user_interaction_filename
from config import test_filename, train_filename, train_test_split_ratio
import pandas as pd
import numpy as np

def preprocess_metadata(metadata_df):
    """
    preprocesses metadata dataframe
    :param metadata_df:
    :return:
    """
    delete_row = metadata_df[metadata_df["reading_time"] == 0].index
    metadata_df = metadata_df.drop(delete_row, inplace=False)
    print("Removed " + str(delete_row.shape[0]) + " rows from metadata df as reading time == 0")
    return metadata_df

def preprocess_user_interaction(df):
    """
    preprocess user interaction data
    :param df: user interaction dataframe
    :return:
    """
    # remove rows with specific condition
    delete_row = df[df["read_percent"] > 100].index
    df.drop(delete_row, inplace=True)
    print("Removed " + str(delete_row.shape[0]) + " rows as read_percent > 100")

    delete_row = df[df["read_percent"] < 0].index
    df.drop(delete_row, inplace=True)
    print("Removed " + str(delete_row.shape[0]) + " rows as read_percent < 10")

    return df

def preprocess_and_split_user_interaction():
    """
    preprocess user interaction data and divides data into train and test
    :return: None
    """
    df = pd.read_csv(user_interaction_filename)
    df['read_percent'] = df['read_percent'].apply(lambda x: int(x))

    # preprocess
    df = preprocess_user_interaction(df)

    # sort the dates
    df.sort_values('updated_at', ascending=False, inplace=True)

    # divide top 25% into test and bottom 25% into train
    test, train = np.split(df, [int(train_test_split_ratio * len(df))])

    test.to_csv(test_filename, sep=',', encoding='utf-8')
    train.to_csv(train_filename, sep=',', encoding='utf-8')