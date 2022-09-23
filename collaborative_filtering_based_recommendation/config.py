train_test_split_ratio = 0.25

# parameters to limit row length
limit_train = True
limit_train_rows = 1000
limit_test = True
limit_test_rows = 10000

# filenames
user_interaction_filename = "data/user-interactions.csv"
test_filename = "data/test.csv"
train_filename = "data/train.csv"
pivot_filename = "data/pivot.pickle"
similarity_filename = "data/similarity.pickle"
recommendation_filename = "data/recommendation.pickle"
train_info_filename = "data/train_info.pickle"

NAN_deafult_value = -1

# recommendation parameters
num_similar_users = 5
num_top_movies_of_similar_users = 5
num_recommend_movies = 5

min_pratilipi_read_for_collabarative = 50

# read write parameters
pivot_file_read_chunk_size = 100