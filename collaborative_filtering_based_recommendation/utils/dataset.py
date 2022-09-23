from config import train_info_filename
import os
import pickle


class dataset_info:
    def __init_(self):
        # length values
        self.unique_users_len = None
        self.unique_pratilipis_len = None

        # conversion dictionaries
        self.user_to_index = {}
        self.index_to_user = {}
        self.pratilipi_to_index = {}
        self.index_to_pratilipi = {}


if os.path.isfile(train_info_filename):
    train_info = dataset_info()
    with open(train_info_filename, "rb") as train_info_file:
        print("loading train info from pickle file")
        train_info = pickle.load(train_info_file)
    test_info = dataset_info()
else:
    train_info = dataset_info()
    test_info = dataset_info()
