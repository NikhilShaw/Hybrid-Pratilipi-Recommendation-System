from utils.preprocess_and_split import preprocess_and_split_user_interaction, preprocess_metadata
from config import train_filename, train_test_split_ratio, popular_pratilipi_filename, limit_train_rows, num_recommend_pratilipis, metadata_filename, num_class_read_time, recommendation_filename, metadata_info_filename, train_info_filename, limit_train
from utils.dataset import train_info, metadata_info, store_info
from utils.popular_movies import store_popular_pratilipis
from utils.table import build_score_matrix, build_pratilipi_feature_matrix, build_user_feature_matrix
from utils.recommendation import store_recommendations
from utils.accuracy import calculate_test_accuracy
from utils.helpers import read_time_to_class

import pandas as pd
import pickle
import numpy as np

# pre process: remove the data
# split the data into train and test
print("[STEP 1/10]: Preprocessing, sorting and splitting into train and test, ratio=" + str(train_test_split_ratio))
preprocess_and_split_user_interaction()

# load the metadata df
print("[STEP 2/10]: Loading metadata df")
metadata_df = pd.read_csv(metadata_filename)

# load the user interaction df
print("[STEP 3/10]: Loading user interaction train df")
train_df = pd.read_csv(train_filename)

# remove rows with readtime == 0
metadata_df = preprocess_metadata(metadata_df)

# get unique authors
metadata_info.unique_authors = list(set(metadata_df["author_id"]))

# get category
metadata_info.unique_category = list(set(metadata_df["category_name"]))

# get pratilipi
metadata_info.unique_pratilipi = list(set(metadata_df["pratilipi_id"]))

# get reading time
reading_times = metadata_df["reading_time"]

# delete the rows in train user_interaction_df which don't have pratilipi id from metadata_df
print("[STEP 4/10]: Finding rows which are in metadata but not in user interactions train")
delete_rows = train_df[~train_df["pratilipi_id"].isin(metadata_info.unique_pratilipi)].index
print("found " + str(len(delete_rows)) + " such rows")
print("deleting the rows")
train_df.drop(delete_rows, inplace=True)


"""
construct an pratilipi vector of following features
[category 1, category 2, ... category 45, below 2 mins, 2 to 10 mins, ..., greater than 120 mins]
total features = 50
"""

# sort the metadata wrt updated_at
print("[STEP 5/10]: Sorting metadata dataframe")
metadata_df.sort_values('updated_at', ascending=False, inplace=True)

# create conversion dictionaries for category
metadata_info.category_id_to_category = {}
metadata_info.category_to_category_id = {}

for index in range(len(metadata_info.unique_category)):
    metadata_info.category_to_category_id[metadata_info.unique_category[index]] = index
    metadata_info.category_id_to_category[index] = metadata_info.unique_category[index]

print("[STEP 6/10]: Iterating over metadata df")
pratilipi_feature_matrix, metadata_info = build_pratilipi_feature_matrix(metadata_df, metadata_info, num_class_read_time, read_time_to_class)

# store popular pratilipis
store_popular_pratilipis(train_df, popular_pratilipi_filename)

# limit the train user interaction
if limit_train is True:
    train_df = train_df.head(limit_train_rows)
else:
    pass

# get unique users
train_info.unique_users = list(set(train_df["user_id"]))

user_feature_matrix, train_info = build_user_feature_matrix(train_df, pratilipi_feature_matrix, metadata_info, train_info, num_class_read_time)

print("[STEP 8/10]: Normalizing user feature matrix")
# divide the user matrix with number of times user is encountered
for index in range(len(train_info.unique_users)):
    user_id = train_info.index_to_user[index]
    pratilipi_read_count = len(train_info.user_to_pratilipis_read[user_id])
    user_feature_matrix[index] /= pratilipi_read_count * 1.0


# get (user, pratilipi) similarity score
print("[STEP 9/10]: Building score matrix")
build_score_matrix(user_feature_matrix, pratilipi_feature_matrix, metadata_info, train_info)

# store train info and metadata info
store_info(train_info, train_info_filename)
store_info(metadata_info, metadata_info_filename)

# calculate accuracy
print("[STEP 10/10]: Calculating accuracy")
average_cover_percentage, weighted_cover_percentage = calculate_test_accuracy()
print("Following are results for test data")
print("")
print("average weighted cover percentage")
print(average_cover_percentage)
print("weighted_cover_percentage")
print(weighted_cover_percentage)














