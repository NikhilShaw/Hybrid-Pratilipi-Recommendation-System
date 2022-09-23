from utils.preprocess_and_split import preprocess_and_split
from config import train_filename, train_test_split_ratio, limit_train_rows, train_info_filename
from utils.dataset import train_info
from utils.table import create_pivot_table, create_similarity_matrix
from utils.recommendation import store_recommendations
from utils.accuracy import calculate_test_accuracy
import pandas as pd
import pickle

# pre process: remove the data
# split the data into train and test
print("[STEP 1]: Preprocessing, sorting and splitting into train and test, ratio=" + str(train_test_split_ratio))
preprocess_and_split()

# load train dataframe
print("[STEP 2]: Loading train data")
train = pd.read_csv(train_filename)

# limit train dataframe
train = train.head(limit_train_rows)

# get unique user list
users = train['user_id']
unique_users = set(users)
train_info.unique_users = unique_users
train_info.unique_users_len = len(unique_users)

# get unique pratilipi list
pratilipis = train['pratilipi_id']
unique_pratilipis = set(pratilipis)
train_info.unique_pratilipis = unique_pratilipis
train_info.unique_pratilipis_len = len(unique_pratilipis)

# create dict for converting user to unique index and vice versa
train_info.user_to_index = {}
train_info.index_to_user = {}

index = 0
for user in unique_users:
    train_info.user_to_index[user] = index
    train_info.index_to_user[index] = user
    index += 1

# create dict for converting pratilipi to unique index and vice versa
train_info.pratilipi_to_index = {}
train_info.index_to_pratilipi = {}

index = 0
for pratilipi in unique_pratilipis:
    train_info.pratilipi_to_index[pratilipi] = index
    train_info.index_to_pratilipi[index] = pratilipi
    index += 1

# build the pivot matrix and store it in external memory
print("[STEP 3]: Creating the pivot matrix and storing in pickle")
create_pivot_table(train, users, unique_users)

# build the similarity matrix and store it in external memory
print("[STEP 4]: Writing in similarity file and stroring it in pickle")
create_similarity_matrix()

# for each user get recommendations
print("[STEP 5]: Calculating and storing recommendation in pickle file")
store_recommendations()

# store train info
with open(train_info_filename, "wb") as train_info_file:
    pickle.dump(train_info, train_info_file)

# test data for accuracy
print("[Step 6/6]: Calculating test accuracy")
average_cover_percentage, weighted_cover_percentage = calculate_test_accuracy()
print("Following is the accuracy of ")
print("Average cover percentage")
print(average_cover_percentage)

print("Average weighted cover percentage")
print(weighted_cover_percentage)

