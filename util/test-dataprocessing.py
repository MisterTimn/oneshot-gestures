import dataprocessing

if __name__=='__main__':
    ds = dataprocessing.DataSaver(('val_loss', 'val_acc', 'dt'))


    for i in xrange(1000000):
        ds.saveValues((i,i*2,i*i))

    ds.saveToCsv("/home/jasper/oneshot-gestures/util","test")
    ds.saveToArray()


    print
