import numpy as np  

def load_DataSet(dataType= "training", path='.' ):
     """

     :param dataType: either training set or testing set
     :param path: curruent directory

     :return:
     """

     data = np.load('ORL_faces.npz')
     if dataType =="training":
        trainX = data['trainX']
        trainY = data['trainY']

        """
        train X = [240, 10304]
        train Y = [240,]
        """
        return trainX,trainY

     elif dataType == "testing":
        testX = data['testX']
        testY = data['testY']

        """
        testX = [160, 10304]
        testY = [160,]
        """
        return testX, testY
     else:
         raise ValueError("dataset must be either 'testing' or 'training'")

#
# a, b  = load_DataSet(dataType='training')
# print(np.shape(a), np.shape(b),b)



 