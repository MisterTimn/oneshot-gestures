import numpy as np
import os

class DataSaver:
    def __init__(self,data_fields):
        self.data_count = 0
        self.capacity = 1000
        self.data_fields = []
        self.data = {}
        try:
            for data_field in data_fields:
                self.data[str(data_field)] = np.empty(self.capacity)
                self.data_fields.append(str(data_field))
        except TypeError:
            self.data[str(data_fields)] = np.empty(self.capacity)
            self.data_fields.append(str(data_fields))

    def saveValues(self,values):
        i = 0
        if (self.data_count >= self.capacity):
            self.doubleCapacity()
        for data_field in self.data_fields:
            self.data[str(data_field)][self.data_count] = values[i]
            i += 1
        self.data_count += 1

    def doubleCapacity(self):
        self.capacity *= 2
        for data_field in self.data_fields:
            self.data[data_field] = np.resize(self.data[data_field],self.capacity)
            print("datasize {}".format(len(self.data[data_field])))

    def saveToCsv(self,path,filename):
        try:
            if path.endswith('/'):
                filepath = "{}{}.csv".format(path,filename)
            else:
                filepath = "{}/{}.csv".format(path,filename)
            file = open(filepath, "w")
            for data_field in self.data_fields:
                file.write("{};".format(data_field))
            file.write("\n")
            for i in xrange(self.data_count):
                for data_field in self.data_fields:
                    file.write("{};".format(self.data[data_field][i]))
                file.write("\n")
        except IOError as e:
            print("I/O error({0}): {1}".format(e.errno, e.strerror))
        except:
            print("Unexpected error")
            raise

    def saveToArray(self,path=None):
        if path == None:
            path = os.path.dirname(os.path.abspath(__file__))
        elif not path.endswith('/'):
            path = "{}/".format(path)
        for data_field in self.data_fields:
            np.save("{}{}".format(path,data_field),np.resize(self.data[data_field],self.data_count+1))

class DataPlotter:
    def __init__(self):
        print

