import pickle

def get_row_from_pickle_file(filename, row_index):
    with open(filename, "rb") as pivot_file:
        for cur_index in range(row_index+1):
            row = pickle.load(pivot_file)
            if cur_index == row_index:
                return row
            else:
                pass
    print("File: " + str(filename) + ", Error row " + str(row_index) + " doesn't exist, max row index is " + str(cur_index))
    return None

def normalize(feature_list):
    print(feature_list.shape)
    len_feature = feature_list.shape[0]

    # get count
    summation = 0
    non_nan_count = 0
    for col_index in range(len_feature):
        if feature_list[col_index] != -1:
            summation += float(feature_list[col_index])
            non_nan_count += 1

    # get mean
    mean = summation*1.0/non_nan_count
    for col_index in range(len_feature):
        if feature_list[col_index] != -1:
            feature_list[col_index] = float(feature_list[col_index]) - mean
        else:
            feature_list[col_index] = mean

    return feature_list
