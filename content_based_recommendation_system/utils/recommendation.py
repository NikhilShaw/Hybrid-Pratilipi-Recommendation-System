from config import *
from utils.helpers import get_row_from_pickle_file
from utils.dataset import train_info
from utils.popular_movies import get_popular_pratilipis
import pickle
import numpy as np


def store_recommendations():
    """
    uses similarity matrix and stores the recommendations in recommendation file

    :return: doesn't return
    """
    metadata_info = None

    with open(recommendation_filename, "wb") as recommendation_file:
        with open(similarity_filename, "rb") as similarity_file:
            for cur_user_index in range(train_info.unique_users_len):

                # get current user similarity vector
                cur_user_similarity = pickle.load(similarity_file)

                # sort similarity in descending order
                cur_user_sorted_indexes = cur_user_similarity.argsort()[::-1]

                # get top similar users
                top_similar_users_index = cur_user_sorted_indexes[:num_similar_users+1]

                # remove current user
                temp_index = np.argwhere(top_similar_users_index == cur_user_index)
                top_similar_users_index = np.delete(top_similar_users_index, temp_index)

                # store the similarity
                top_similar_user_similarity = cur_user_similarity[top_similar_users_index]

                # get the indexes of watched movies by current user
                cur_user_watched_movies_index = []
                cur_user_scores = get_row_from_pickle_file(pivot_filename, cur_user_index)
                for index in range(train_info.unique_pratilipis_len):
                    if cur_user_scores[index] != NAN_deafult_value:
                        cur_user_watched_movies_index.append(index)

                # get top movies of each top similar user
                top_score_index_list = []
                weight_list = []

                # print("top similar users index")
                # print(top_similar_users_index)
                # print("top similar users similartiy")
                # print(top_similar_user_similarity)

                for similar_user_index in top_similar_users_index:
                    # get the score of similar user
                    similar_user_scores = get_row_from_pickle_file(pivot_filename, similar_user_index)

                    # sort the score in descending order
                    similar_user_sorted_scores_index = similar_user_scores.argsort()[::-1]

                    # get top pratilipis of similar users that are not watched by current user
                    similar_user_top_scores_index = []
                    for index in similar_user_sorted_scores_index:
                        if index not in cur_user_watched_movies_index:
                            similar_user_top_scores_index.append(index)
                            if len(similar_user_top_scores_index) == num_top_pratilipi_of_similar_users:
                                break

                    similar_user_top_scores_index = np.array(similar_user_top_scores_index)
                    similar_user_top_scores = similar_user_scores[similar_user_top_scores_index]

                    # print("top user pratilipi score")
                    # print(similar_user_top_scores)
                    # multiply the score with similarity
                    similar_user_weights = similar_user_top_scores * cur_user_similarity[similar_user_index]

                    for index in similar_user_top_scores_index:
                        top_score_index_list.append(index)

                    for w in similar_user_weights:
                        weight_list.append(w)

                # recommend top num_recommend_movies
                weight_list = np.array(weight_list)
                # print("weight list")
                # print(weight_list)
                top_weight_index = weight_list.argsort()[::-1][:num_recommend_pratilipis]

                # print("top weights")
                # print(weight_list[top_weight_index])

                # store the top movies index in pickle file
                top_pratilipis = [metadata_info.index_to_pratilipi[x] for x in top_weight_index]

                # print("top pratilipis")
                # print(top_pratilipis)
                pickle.dump(top_weight_index, recommendation_file)
                # print("")
                # print("-----------")
                # print("")

def get_recommendation(user_list):
    """
    Gets recommendation for username(s)

    :param user_list: list of user names
    :return: list of list of recommended pratilipi names
    """
    recommendations = []
    popular_pratilipis = get_popular_pratilipis(popular_pratilipi_filename)
    for user_name in user_list:
        # if its an old user
        if user_name in train_info.unique_users:
            # get user index
            user_index = train_info.user_to_index[user_name]
            # get the recommended pratilipis id
            recommended_pratilipis_indexes = get_row_from_pickle_file(recommendation_filename, user_index)
            # get the name of the pratilipis
            top_pratilipis = [x for x in recommended_pratilipis_indexes]
        else:
            # if its a new user recommend popular movies
            top_pratilipis = popular_pratilipis

        recommendations.append(top_pratilipis)

    return recommendations