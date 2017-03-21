import dataprocessing
import numpy as np
if __name__=='__main__':
    dp = dataprocessing.DataPlotter()

    # losses = []
    # val_accuracies = []
    # train_losses=[]
    # for num_samples in [200,100,50]:
    #     losses.append(np.load("/home/jasper/oneshot-gestures/output/data-17-retrain-1-samples-{}/train_loss.npy".format(num_samples)))
    #     losses.append(np.load("/home/jasper/oneshot-gestures/output/data-17-retrain-1-samples-{}/val_loss.npy".format(num_samples)))
    #     val_accuracies.append(np.load("/home/jasper/oneshot-gestures/output/data-17-retrain-1-samples-{}/val_acc.npy".format(num_samples)))


    path = "/home/jasper/oneshot-gestures/output/"
    losses = np.empty((3,5),dtype=object)
    val_accuracies = np.empty((3,5),dtype=object)
    class_accuracies = np.empty((3,5),dtype=object)


    # num_class = 15
    # j=0
    # for num_layers in [2,3]:
    #     i=0
    #     for num_samples in [200,100,50,25,10]:
    #         losses[j][i]=np.load("{}data-{}-retrain-{}-samples-{}/val_loss.npy".format(path,num_class,num_layers,num_samples))
    #         val_accuracies[j][i]=np.load("{}data-{}-retrain-{}-samples-{}/val_acc.npy".format(path, num_class, num_layers, num_samples))
    #         class_accuracies[j][i]=np.load("{}data-{}-retrain-{}-samples-{}/class_acc.npy".format(path, num_class, num_layers, num_samples))
    #         i+=1
    #     j+=1
    #
    # for i in range(len(class_accuracies[0])):
    #     print(len(class_accuracies[0][i]))
    #     dp.plotCompare((class_accuracies[0][i],class_accuracies[1][i]))

    num_class = 15
    """
    10  -> 500
    25  -> 400
    50  -> 300
    100 -> 100
    """
    data = []
    for retrain_layers in [2,3]:
        for num_samples in [1,25,200]:
            data.append(np.load("{}data-{}-retrain-{}-samples-{}/class_acc.npy".format(path, num_class, retrain_layers, num_samples)))

        dp.plotCompare(data)

        #dp.plotAccLoss(val_loss,val_acc)