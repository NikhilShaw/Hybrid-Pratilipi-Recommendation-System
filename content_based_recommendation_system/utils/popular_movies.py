from config import num_recommend_pratilipis
import pickle
import os
import numpy as np

def get_popular_pratilipis(popular_pratilipi_filename):
    # if its already stored then return it
    if os.path.isfile(popular_pratilipi_filename):
        with open(popular_pratilipi_filename, "rb") as popular_pratilipi_file:
            print("loading popular pratilipis from pickle file")
            popular_pratilipis = pickle.load(popular_pratilipi_file)
            return popular_pratilipis
    else:
        print("Error: Can't find poupular pratilipi file, call utils.popular_moveis.store_popular_pratilipis first")
        return None

def store_popular_pratilipis(train_df, popular_pratilipi_filename):

    # if its already stored then do nothing
    if os.path.isfile(popular_pratilipi_filename):
        pass
    else:
        # calculate count of each pratilipi
        pratilipi_to_index = {}
        index_to_pratilipi = {}
        unique_pratilipi_set = list(set(train_df["pratilipi_id"]))

        count = 0
        for pratilipi in unique_pratilipi_set:
            pratilipi_to_index[pratilipi] = count
            index_to_pratilipi[count] = pratilipi
            count += 1

        pratilipi_count = [0 for x in range(len(unique_pratilipi_set))]
        pratilipis = train_df["pratilipi_id"]

        for pratilipi in pratilipis:
            index = pratilipi_to_index[pratilipi]
            pratilipi_count[index] += 1

        pratilipi_count = np.array(pratilipi_count)

        pratilipi_max_count_index = pratilipi_count.argsort()[::-1][:num_recommend_pratilipis]

        # get top pratilipi movies
        popular_pratilipis = []
        for index in pratilipi_max_count_index:
            popular_pratilipis.append(index_to_pratilipi[index])

        # store it
        with open(popular_pratilipi_filename, "wb") as popular_pratilipi_filename:
            pickle.dump(popular_pratilipis, popular_pratilipi_filename)





