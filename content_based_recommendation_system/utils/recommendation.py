from config import *
from utils.helpers import get_row_from_pickle_file
from utils.dataset import train_info
from utils.popular_movies import get_popular_pratilipis


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