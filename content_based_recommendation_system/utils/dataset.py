from config import train_info_filename, metadata_info_filename
import os
import pickle


class user_interaction_info:
    def __init_(self):
        # length values
        self.unique_users = None
        self.unique_pratilipis_len = None

        # conversion dictionaries
        self.user_to_index = {}
        self.index_to_user = {}
        self.pratilipi_to_index = {}
        self.index_to_pratilipi = {}


class metadata_info_class:
    def __init_(self):
        self.unique_authors = None
        self.unique_category = None
        self.unique_pratilipi = None


if os.path.isfile(train_info_filename):
    with open(train_info_filename, "rb") as train_info_file:
        print("loading train info from pickle file")
        train_info = pickle.load(train_info_file)

else:
    train_info = user_interaction_info()

if os.path.isfile(metadata_info_filename):
    with open(metadata_info_filename, "rb") as metadata_info_file:
        print("loading metadata info from pickle file")
        metadata_info = pickle.load(metadata_info_file)
else:
    metadata_info = metadata_info_class()

test_info = user_interaction_info()
