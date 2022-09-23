from config import user_interaction_filename
from config import test_filename, train_filename, train_test_split_ratio, min_pratilipi_read_for_collabarative
import pandas as pd
import numpy as np

def preprocess(df):
    # remove rows with specific condition
    delete_row = df[df["read_percent"] > 100].index
    df.drop(delete_row, inplace=True)
    print("Removed " + str(delete_row.shape[0]) + " rows as read_percent > 100")

    delete_row = df[df["read_percent"] < 0].index
    df.drop(delete_row, inplace=True)
    print("Removed " + str(delete_row.shape[0]) + " rows as read_percent < 10")

    return df

def div_train_dataset(train):
    """
    used to divide train into two parts based on config.min_pratilipi_read_for_collabarative
    :param train: train dataset
    :return: train_collaborative, train_content
    """

    user_list = train['user_id']

    # get unique users
    unique_users = set(user_list)

    # calculate books read by each user
    books_read = {}
    for user in unique_users:
        books_read[user] = 0

    for user in user_list:
        books_read[user] += 1

    # divide users based on threshold of pratilipis read
    high_read_users = []
    low_read_users = []
    for user in unique_users:
        if books_read[user] >= min_pratilipi_read_for_collabarative:
            high_read_users.append(user)
        else:
            low_read_users.append(user)

    print("high read users length")
    print(len(high_read_users))
    print("low read users length")
    print(len(low_read_users))

    high_read_users_rows_ids = train[train["user_id"].isin(high_read_users)].index
    low_read_users_rows_ids = train[train["user_id"].isin(low_read_users)].index

    train_collaborative = train.copy()
    train_collaborative.drop(low_read_users_rows_ids, inplace=True)

    train_content = train
    train_content.drop(high_read_users_rows_ids, inplace=True)

    print("collab train length")
    print(len(train_collaborative.index))

    print("content train length")
    print(len(train_content.index))

    return train_collaborative, train_content

def preprocess_and_split():
    df = pd.read_csv(user_interaction_filename)
    df['read_percent'] = df['read_percent'].apply(lambda x: int(x))

    df = preprocess(df)

    # sort the dates
    df.sort_values('updated_at', ascending=False, inplace=True)

    # divide top 25% into test and bottom 25% into train
    test, train = np.split(df, [int(train_test_split_ratio * len(df))])

    test.to_csv(test_filename, sep=',', encoding='utf-8')
    train.to_csv(train_filename, sep=',', encoding='utf-8')