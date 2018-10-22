import os
import csv
import h5py
import numpy as np

class Path():
    def __init__(self):
        self.resources = os.path.join("..", "data")
        self.results = os.path.join("..", "out")

    def getFileDir(self, file_name):
        return os.path.join(self.resources, file_name)

    def getNameResult(self, file_name, extension):
        if (extension is None):
            return file_name
        else:
            try:
                splitted = file_name.split(".")
                return splitted[0] + "_" + extension + "." + splitted[1]
            except:
                return file_name + "_" + extension 

    def getPathSave(self, name):
        os.makedirs(self.results, exist_ok=True)
        return os.path.join(self.results, name)

class Data():
    def saveVariable(self, name, extension, value):
        n = Path().getNameResult(name+".txt", extension)
        with open(Path().getPathSave(n), "w") as variable_file:
            variable_file.write(value)

    def fetchFromCSV(self, file_name):
        reader = csv.reader(open(Path().getFileDir(file_name), 'rt'))
        return [[Utils().convertTypes(item) for item in row] for row in reader]

    def fetchFromH5(self, train_name, test_name):
        train_dataset = h5py.File(Path().getFileDir(train_name), "r")
        train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
        train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

        test_dataset = h5py.File(Path().getFileDir(test_name), "r")
        test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
        test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

        classes = np.array(test_dataset["list_classes"][:]) # the list of classes
        
        train_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T # The "-1" makes reshape flatten the remaining dimensions
        test_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T    # Standardize data to have feature values between 0 and 1.

        train_x = train_x_flatten/255.
        test_x = test_x_flatten/255.

        return train_x.transpose(), train_set_y_orig, test_x.transpose(), test_set_y_orig, classes

class Utils():
    def convertTypes(self, s):
        s = s.strip()
        try:
            return float(s) if '.' in s else int(s)
        except ValueError:
            return s	