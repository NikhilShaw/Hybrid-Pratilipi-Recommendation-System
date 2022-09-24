from config import test_filename, num_recommend_pratilipis, limit_test_rows, limit_test
import pandas as pd
from utils.recommendation import get_recommendation


class accuracy_metrics:
    def __init__(self):
        self.cover_percentage = None
        self.weighted_cover_percentage = None
        self.total_test_movies_watched = 0
        self.weighted_cover = 0
        self.cover_count = 0
        self.num_recommend_movies = num_recommend_pratilipis


def calculate_accuracy(user_name_list, recommended_movies_list):
    # load test dataframe
    print("loading test data")
    test = pd.read_csv(test_filename)
    if limit_test is True:
        test = test.head(limit_test_rows)
    else:
        pass

    # initialize accuracy metrics
    accuracy_metrics_list = []
    for x in range(len(user_name_list)):
        accuracy_metrics_obj = accuracy_metrics()
        accuracy_metrics_list.append(accuracy_metrics_obj)

    # calculate the accuracy by iterating over test df
    print("iterating over test data")
    for index, row in test.iterrows():
        match_index = None
        try:
            match_index = user_name_list.index(row["user_id"])
        except ValueError as e:
            pass
        if match_index is not None:
            accuracy_metrics_list[match_index].total_test_movies_watched += 1
            if row["pratilipi_id"] in recommended_movies_list[match_index]:
                accuracy_metrics_list[match_index].cover_count += 1
                accuracy_metrics_list[match_index].weighted_cover += float(row["read_percent"])/100.0
            else:
                pass
        else:
            pass

    for accuracy_metrics_obj in accuracy_metrics_list:
        if accuracy_metrics_obj.total_test_movies_watched > accuracy_metrics_obj.num_recommend_movies:
            accuracy_metrics_obj.cover_percentage = (accuracy_metrics_obj.cover_count*1.0/accuracy_metrics_obj.num_recommend_movies)*100
            accuracy_metrics_obj.weighted_cover_percentage = (accuracy_metrics_obj.weighted_cover*1.0/accuracy_metrics_obj.num_recommend_movies)*100
        else:
            accuracy_metrics_obj.cover_percentage = (accuracy_metrics_obj.cover_count*1.0/accuracy_metrics_obj.total_test_movies_watched)*100
            accuracy_metrics_obj.weighted_cover_percentage = (accuracy_metrics_obj.weighted_cover*1.0/accuracy_metrics_obj.total_test_movies_watched)*100

    return accuracy_metrics_list


def calculate_test_accuracy():
    print("loading test data")
    test_df = pd.read_csv(test_filename)
    if limit_test is True:
        test_df = test_df.head(limit_test_rows)
    else:
        pass

    # get unique test users
    test_unique_users = list(set(test_df["user_id"]))

    # get recommendations for the users
    recomended_pratilipis = get_recommendation(test_unique_users)

    print("test users len")
    print(len(test_unique_users))

    print("test recommended pratilipis len")
    print(len(recomended_pratilipis))

    accuracy_metrics_list = calculate_accuracy(test_unique_users, recomended_pratilipis)

    average_cover_percentage = 0
    weighted_cover_percentage = 0
    for accuracy_metrics_obj in accuracy_metrics_list:
        average_cover_percentage += accuracy_metrics_obj.cover_percentage
        weighted_cover_percentage += accuracy_metrics_obj.weighted_cover_percentage

    average_cover_percentage /= len(accuracy_metrics_list)
    weighted_cover_percentage /= len(accuracy_metrics_list)

    return average_cover_percentage, weighted_cover_percentage






