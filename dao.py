from PIL import Image
from os import walk
from os import environ
import pickle

STORAGE_PATH = environ.get("STORAGE_PATH")
FIT_DATA_PATH = STORAGE_PATH + '/fit_data.pckl'

PRE_FIT_DATE_PATH = 'pre_fit'


def get_pre_fit_data():
    data = []
    for (dir_path, dir_names, file_names) in walk(PRE_FIT_DATE_PATH):
        for dir_name in dir_names:
            print(dir_name)

            for (fig_path, fig_dirs, figs) in walk(dir_path + '/' + dir_name):
                for fig in figs:
                    image = Image.open(fig_path + '/' + fig)
                    data.append((image, dir_name))

    return data


def get_fit_data():
    with open(FIT_DATA_PATH, 'rb') as file:
        print('Get fit data from', FIT_DATA_PATH)
        return pickle.load(file)


def save_fit_data(fit_data):
    with open(FIT_DATA_PATH, 'wb') as file:
        pickle.dump(fit_data, file)
        print('Fit data saved in', FIT_DATA_PATH)
