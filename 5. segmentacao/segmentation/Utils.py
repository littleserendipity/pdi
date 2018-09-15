import os

class Path():
    def __init__(self):
        self.resources = os.path.join("..", "images")
        self.results = os.path.join("results")

    def getFileDir(self, file_name):
        return os.path.join(self.resources, file_name)

    def getNameResult(self, file_name, extension):
        return (extension + "_" + file_name)

    def getPathSave(self, name):
        os.makedirs(self.results, exist_ok=True)
        return os.path.join(self.results, name)