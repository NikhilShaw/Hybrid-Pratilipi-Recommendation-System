import pickle

def read_time_to_class(read_time):
    if 0 < read_time < 120:
        return 0

    elif 120 <= read_time < 10*60:
        return 1

    elif 10*60 <= read_time < 30*60:
        return 2

    elif 30*60 <= read_time < 120*60:
        return 3

    elif read_time >= 120*60:
        return 4
    else:
        print(type(read_time))
        print("ERROR, read time is "+ str(read_time))
        return None


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
