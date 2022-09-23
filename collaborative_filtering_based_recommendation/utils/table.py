from config import pivot_filename, similarity_filename, NAN_deafult_value, pivot_file_read_chunk_size
from utils.dataset import train_info
from sklearn.metrics.pairwise import cosine_similarity
from utils.helpers import get_row_from_pickle_file
import pickle
import numpy as np

def create_user_to_read_pratilipis(users, unique_users, train):
    # store key(user) and val(list of (pratilipis, read_percentage) read by user] in a dict
    user_to_read_pratilipis = {}
    for user in unique_users:
        user_to_read_pratilipis[user] = []

    user_len = len(users)
    count = 0
    for train_index, train_row in train.iterrows():
        print("storing in tree: " + str(count) + " / " + str(user_len))
        key = train_row['user_id']
        val = (train_row['pratilipi_id'], train_row['read_percent'])
        user_to_read_pratilipis[key].append(val)
        count += 1
    return user_to_read_pratilipis

def create_pivot_table(train, users, unique_users):
    # get user( as key) to (pratilipi, read_percentage) (as val)
    user_to_read_pratilipis = create_user_to_read_pratilipis(users, unique_users, train)

    # create the pivot table
    count = 0
    with open(pivot_filename, "wb") as pivot_file:
        for user in unique_users:
            print("[STEP 3]: Creating the pivot matrix and storing in pickle: " + str(count) + " / " + str(len(unique_users)))
            csv_row = [NAN_deafult_value for x in range(train_info.unique_pratilipis_len)]
            pratilipis_read = user_to_read_pratilipis[user]
            if pratilipis_read is None or len(pratilipis_read) == 0:
                print("Error: No pratilipi read")
            else:
                for pratilipi_and_read_percent in pratilipis_read:
                    col_index = train_info.pratilipi_to_index[pratilipi_and_read_percent[0]]
                    csv_row[col_index] = pratilipi_and_read_percent[1]
            pickle.dump(np.array(csv_row), pivot_file)
            count += 1

def create_similarity_matrix():
    count = 0
    with open(similarity_filename, "wb") as similarity_file:
        for cur_index in range(train_info.unique_users_len):

            # get the current row
            cur_feature = get_row_from_pickle_file(pivot_filename, cur_index)

            # compare similarity with all other rows
            with open(pivot_filename, "rb") as pivot_file:
                print("[STEP 4]: Writing in similarity file and stroring it in pickle: " + str(count) + " / " + str(train_info.unique_users_len))
                cur_similarity = []
                index = 0
                while True:
                    chunk = []
                    for chunk_index in range(pivot_file_read_chunk_size):
                        if index == train_info.unique_users_len:
                            break
                        else:
                            feature = pickle.load(pivot_file)
                            chunk.append(feature)
                            index += 1

                    # end of file and nothing is in chunk
                    if len(chunk) != 0:
                        chunk = np.array(chunk)
                        mat = np.vstack((cur_feature, chunk))
                        similarities = cosine_similarity(mat)

                        for similarity in similarities[0][1:]:
                            cur_similarity.append(similarity)
                    else:
                        break

                cur_similarity = np.array(cur_similarity)

                pickle.dump(cur_similarity, similarity_file)
                count += 1

