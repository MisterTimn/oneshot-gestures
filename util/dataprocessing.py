import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.lines as mlines
import numpy as np
import os
import itertools

from sklearn import metrics


class DataSaver:
    def __init__(self,data_fields):
        self.data_count = 0
        self.capacity = 1000
        self.data_fields = []
        self.data = {}
        try:
            for data_field in data_fields:
                self.data[str(data_field)] = np.empty(self.capacity,dtype='float64')
                self.data_fields.append(str(data_field))
        except TypeError:
            self.data[str(data_fields)] = np.empty(self.capacity,dtype='float64')
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
                    file.write("{};".format(self.data[data_field][i]).replace('.',','))
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
            np.save("{}{}".format(path,data_field),np.resize(self.data[data_field],self.data_count))

class DataPlotter:
    def __init__(self):
        print
        mpl.rcParams['pgf.rcfonts']=False
        blue_line =     mlines.Line2D([],[],color='blue', marker='o')
        red_line =      mlines.Line2D([],[],color='red', marker='o')
        green_line =    mlines.Line2D([],[],color='green', marker='o')
        yellow_line =   mlines.Line2D([],[], color='yellow', marker='o')
        self.lines=(blue_line,red_line,green_line,yellow_line)
        # mpl.rcParams['pgf.texsystem'] = u'xelatex',  # change this if using xetex or lautex


    def plotAccLoss(self,loss,acc):
        plt.figure(1)

        plt.subplot(121)
        lines = plt.plot(loss)
        plt.grid(True)
        plt.ylabel('val_loss')
        plt.setp(lines, color='r', linewidth=1.0)

        plt.subplot(122)
        lines2 = plt.plot(acc)
        plt.grid(True)
        plt.ylabel('val_acc')
        plt.setp(lines2, color='b', linewidth=1.0)

        plt.show()

    def plotAccF1(self,y_test,y_predictions,x_labels,oneshot_class,title="F1 score"):
        plt.grid(True)
        f1_weighted = np.zeros(len(y_predictions))
        f1_class = np.zeros(len(y_predictions))
        i=0
        for y_pred in y_predictions:
            f1_scores = metrics.f1_score(y_test,y_pred,average=None)
            f1_class[i] = f1_scores[oneshot_class]
            f1_weighted[i] = metrics.f1_score(y_test,y_pred,average='weighted')
            # accuracies[i] = metrics.accuracy_score(y_test,y_pred)
            i+=1

        plt.xticks(np.arange(len(x_labels)),x_labels)
        plt.xlabel("Number of samples")
        plt.ylabel("F1 score")

        plt.plot(f1_weighted,'b')
        plt.plot(f1_class,'r')

        plt.plot(f1_weighted, 'bo')
        plt.plot(f1_class, 'ro')

        blue_line =     mlines.Line2D([],[],color='blue', marker='o',label='Weighted F1-score')
        red_line =      mlines.Line2D([],[],color='red', marker='o',label="F1-score oneshot-class")

        plt.legend(handles=(blue_line,red_line))

        ax=plt.subplot(111)
        ax.set_xscale('log')
        plt.title(title)
        plt.show()

    def plotDoubleClassF1(self,y_test,y_predictions,x_labels,oneshot_class,oneshot_class_2,title="F1 score"):
        plt.grid(True)
        f1_weighted = np.zeros(len(y_predictions))
        f1_class_1 = np.zeros(len(y_predictions))
        f1_class_2 = np.zeros(len(y_predictions))
        i=0
        for y_pred in y_predictions:
            f1_scores = metrics.f1_score(y_test,y_pred,average=None)
            f1_class_1[i] = f1_scores[oneshot_class]
            f1_class_2[i] = f1_scores[oneshot_class_2]
            f1_weighted[i] = metrics.f1_score(y_test,y_pred,average='weighted')
            # accuracies[i] = metrics.accuracy_score(y_test,y_pred)
            i+=1

        plt.xticks(np.arange(len(x_labels)),x_labels)
        plt.xlabel("Number of samples")
        plt.ylabel("F1 score")

        plt.plot(f1_weighted,'b')
        plt.plot(f1_class_1,'r')
        plt.plot(f1_class_2, 'g')

        plt.plot(f1_weighted, 'bo')
        plt.plot(f1_class_1, 'ro')
        plt.plot(f1_class_2, 'go')

        blue_line =     mlines.Line2D([],[],color='blue', marker='o',label='Weighted F1-score')
        red_line =      mlines.Line2D([],[],color='red', marker='o',label="F1-score gesture {}".format(14))
        green_line =    mlines.Line2D([],[],color='green', marker='o',label="F1-score gesture {}".format(15))

        plt.legend(handles=(blue_line,red_line,green_line))
        plt.title(title)
        plt.show()

    def plot_confusion_matrix(self,cm, classes,
                              normalize=False,
                              title='Confusion matrix',
                              cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
            thresh = cm.max() / 2.
            for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                plt.text(j, i, "{:2.0f}".format(cm[i, j]*100.0),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")
        else:
            print('Confusion matrix, without normalization')
            thresh = cm.max() / 2.
            for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                plt.text(j, i, cm[i, j],
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")

        # plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

    def plotConfusionMatrix(self, y_test, y_pred,title="Confusion Matrix", savePath=None):
        class_names = ["0"]
        for i in xrange(1,20):
            class_names.append("{}".format(i))
        print(class_names)

        cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
        np.set_printoptions(precision=2)


        # Plot non-normalized confusion matrix
        plt.figure(figsize=(8,6))
        self.plot_confusion_matrix(cnf_matrix, classes=class_names,
                              title=title)
        if savePath != None:
            # plt.savefig("{}.pgf".format(savePath))
            plt.savefig("{}.pdf".format(savePath))

        # Plot normalized confusion matrix
        plt.figure(figsize=(8,6))
        self.plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                              title="{} normalized".format(title))
        if savePath != None:
            # plt.savefig("{}-norm.pgf".format(savePath))
            plt.savefig("{}-norm.pdf".format(savePath))

        # plt.show()




