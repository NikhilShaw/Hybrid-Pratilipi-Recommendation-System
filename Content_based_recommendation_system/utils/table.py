from config import recommendation_filename, num_recommend_pratilipis
import pickle
import numpy as np

def build_score_matrix(user_feature_matrix, pratilipi_feature_matrix, metadata_info, train_info):
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

def build_pratilipi_feature_matrix(metadata_df, metadata_info, num_class_read_time, read_time_to_class):
    # create conversion dictionaries for pratilipi of metadata df
    metadata_info.pratilipi_to_index = {}
    metadata_info.index_to_pratilipi = {}

    # create pratilipi feature matrix
    pratilipi_feature_matrix = []

    # keep count of encountered pratilipis
    encountered_pratilipis = set()
    unique_pratilipi_count = 0

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

        return pratilipi_feature_matrix, metadata_info

def build_user_feature_matrix(train_df, pratilipi_feature_matrix, metadata_info, train_info, num_class_read_time):
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
        user_feature = pratilipi_feature * (row["read_percent"] / 100.0)
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

    return user_feature_matrix, train_info