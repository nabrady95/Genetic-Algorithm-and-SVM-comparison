import numpy
from GA import geneticAlgorithm
from datasetLoading import loadBreastCancer, loadCreditCard
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
        
dataInputs, dataOutputs = loadCreditCard()
dataInputs = numpy.array(dataInputs)
dataOutputs = numpy.array(dataOutputs)
noOfClasses = len(numpy.unique(dataOutputs))
neuronPerLayer = int(len(dataInputs[0])/3)

std = StandardScaler()
kf = KFold(n_splits=5)
print(kf.get_n_splits(dataInputs))

kFoldAccuracies=[]
kFoldTimes = []
iteration=0

from tqdm import tqdm
pbar = tqdm(total=5)

for train_index , test_index in kf.split(dataInputs):
    X_train, X_test= dataInputs[train_index], dataInputs[test_index]
    y_train, y_test= dataOutputs[train_index], dataOutputs[test_index]
    #Transform data 
    X_train = std.fit_transform(X_train)
    X_test = std.transform(X_test)
    acc, network = geneticAlgorithm(8, 0.527, 0.15, [neuronPerLayer,neuronPerLayer,noOfClasses], 5, X_train, y_train, False, False, False, 0)
    accuracyTest = network.evaluate(X_test, y_test)
    kFoldAccuracies.append(accuracyTest)
    pbar.update(1)

meanAccuracy = sum(kFoldAccuracies)/len(kFoldAccuracies)  
print("")
print ("Mean testing Accuracy: ", meanAccuracy)