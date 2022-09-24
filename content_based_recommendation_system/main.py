from utils.preprocess_and_split import preprocess_and_split, preprocess_metadata
from config import train_filename, train_test_split_ratio, popular_pratilipi_filename, limit_train_rows, num_recommend_pratilipis, metadata_filename, num_class_read_time, recommendation_filename, metadata_info_filename, train_info_filename, limit_train
from utils.dataset import train_info, metadata_info
from utils.popular_movies import store_popular_pratilipis
from utils.accuracy import calculate_test_accuracy
from utils.helpers import read_time_to_class

import pandas as pd
import pickle
import numpy as np

# pre process: remove the data
# split the data into train and test
print("[STEP 1/10]: Preprocessing, sorting and splitting into train and test, ratio=" + str(train_test_split_ratio))
preprocess_and_split()

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


# create conversion dictionaries for pratilipi of metadata df
metadata_info.pratilipi_to_index = {}
metadata_info.index_to_pratilipi = {}

# create pratilipi feature matrix
pratilipi_feature_matrix = []

# keep count of encountered pratilipis
encountered_pratilipis = set()
unique_pratilipi_count = 0

print("[STEP 6/10]: Iterating over metadata df")
count = 0
for index, row in metadata_df.iterrows():
    count += 1
    print("[STEP 6/10]: Iterating metadata df " + str(count) + " / " + str(len(metadata_df.index)))
    # check if already encountered
    if row["pratilipi_id"] not in encountered_pratilipis:

        # if new pratilipi encountered then store it in set for faster search
        encountered_pratilipis.add(row["pratilipi_id"])

        # create one hot encoding of category
        category_feature = np.array([0 for x in range(len(metadata_info.unique_category))])
        category_index = metadata_info.category_to_category_id[row["category_name"]]
        category_feature[category_index] = 1

        # create one hot encoding of read_time class
        read_time_feature = np.zeros(num_class_read_time)
        read_index = read_time_to_class(row["reading_time"])
        read_time_feature[read_index] = 1

        # create pratilipi feature
        pratilipi_feature = np.concatenate([category_feature, read_time_feature])

        # store it in pratilipi matrix
        pratilipi_feature_matrix.append(pratilipi_feature)

        # update conversion dictionaries
        metadata_info.pratilipi_to_index[row["pratilipi_id"]] = unique_pratilipi_count
        metadata_info.index_to_pratilipi[unique_pratilipi_count] = row["pratilipi_id"]

        unique_pratilipi_count += 1

    else:
        pass

pratilipi_feature_matrix = np.array(pratilipi_feature_matrix)

if unique_pratilipi_count != pratilipi_feature_matrix.shape[0]:
    print("Error unique pratilipi count != pratilipi_feature_matrix rows")
else:
    pass

if unique_pratilipi_count != len(metadata_info.unique_pratilipi):
    print("Error unique pratilipi count != stored unique pratilipi count")
else:
    pass

# store popular pratilipis
store_popular_pratilipis(train_df, popular_pratilipi_filename)

# limit the train user interaction
if limit_train is True:
    train_df = train_df.head(limit_train_rows)
else:
    pass

# get unique users
train_info.unique_users = list(set(train_df["user_id"]))

# create user feature matrix
feature_len = len(metadata_info.unique_category) + num_class_read_time
zero_feature = [0 for x in range(feature_len)]
user_feature_matrix = [zero_feature for x in range(len(train_info.unique_users))]
user_feature_matrix = np.array(user_feature_matrix)
user_feature_matrix = user_feature_matrix.astype('float')


# create user conversion dictionaries
train_info.user_to_index = {}
train_info.index_to_user = {}

# create a set to find if user already encountered
user_encountered = set()

# create user to pratilipis read
train_info.user_to_pratilipis_read = {}

print("[STEP 7/10]: Iterating over train df")
unique_user_count = 0
count = 0
for index, row in train_df.iterrows():
    count += 1
    print("[STEP 7/10]: Iterating user interaction train df " + str(count) + " / " + str(len(train_df.index)))

    # get pratilipi index and its feature
    pratilipi_index = metadata_info.pratilipi_to_index[row["pratilipi_id"]]
    pratilipi_feature = pratilipi_feature_matrix[pratilipi_index]

    # check if user already encountered
    if row["user_id"] not in user_encountered:

        # if not then add to conversion dictionary
        user_encountered.add(row["user_id"])

        # update the conversion dictionaries
        train_info.index_to_user[unique_user_count] = row["user_id"]
        train_info.user_to_index[row["user_id"]] = unique_user_count

        # add to the pratilipis read dictionary
        train_info.user_to_pratilipis_read[row["user_id"]] = [row["pratilipi_id"]]

        unique_user_count += 1

    else:
        pass

    # calculate the user feature matrix = read_percentage * pratilipi_feature
    user_feature = pratilipi_feature * (row["read_percent"]/100.0)
    user_index = train_info.user_to_index[row["user_id"]]
    user_feature_matrix[user_index] += user_feature

    # update the pratilipis read dictionary
    train_info.user_to_pratilipis_read[row["user_id"]].append(row["pratilipi_id"])

if unique_user_count != user_feature_matrix.shape[0]:
    print("unique_user_count != user_feature_matrix.shape[0]")
else:
    pass

if unique_user_count != len(train_info.unique_users):
    print(" unique user count != len(train_info.unique_users)")
else:
    pass

print("[STEP 8/10]: Normalizing user feature matrix")
# divide the user matrix with number of times user is encountered
for index in range(len(train_info.unique_users)):
    user_id = train_info.index_to_user[index]
    pratilipi_read_count = len(train_info.user_to_pratilipis_read[user_id])
    user_feature_matrix[index] /= pratilipi_read_count * 1.0


# get (user, pratilipi) similarity score
print("[STEP 9/10]: Building score matrix")
count = 0
with open(recommendation_filename, "wb") as recommendation_file:
    for user_index in range(len(user_feature_matrix)):
        count += 1
        print("[STEP 9/10]: Storing recommendation " + str(count) + " / " + str(len(user_feature_matrix)))

        # transpose
        pratilipi_feature_matrix_transpose = np.transpose(pratilipi_feature_matrix)

        # get user feature
        user_feature = user_feature_matrix[user_index]

        # get similarity score with each pratilipi
        similarity_scores = np.matmul(user_feature, pratilipi_feature_matrix_transpose)

        # arg sort the scores
        similarity_scores_sorted_indexes = similarity_scores.argsort()[::-1]

        # get pratilipis watched by user
        user_id = train_info.index_to_user[user_index]
        pratilipis_read = train_info.user_to_pratilipis_read[user_id]
        pratilipis_read_index = [metadata_info.pratilipi_to_index[x] for x in pratilipis_read]

        # get top n pratilipis not watched by user
        top_pratilipi = []

        for similarity_index in similarity_scores_sorted_indexes:
            if similarity_index not in pratilipis_read_index:
                top_pratilipi.append(metadata_info.index_to_pratilipi[similarity_index])
                if len(top_pratilipi) == num_recommend_pratilipis:
                    break

        # store it in pickle file
        pickle.dump(top_pratilipi, recommendation_file)

# store train info and metadata info
with open(train_info_filename, "wb") as train_info_file:
    pickle.dump(train_info, train_info_file)

with open(metadata_info_filename, "wb") as metadata_info_file:
    pickle.dump(metadata_info, metadata_info_file)

# calculate accuracy
print("[STEP 10/10]: Calculating accuracy")
average_cover_percentage, weighted_cover_percentage = calculate_test_accuracy()
print("Following are results for test data")
print("")
print("average weighted cover percentage")
print(average_cover_percentage)
print("weighted_cover_percentage")
print(weighted_cover_percentage)














