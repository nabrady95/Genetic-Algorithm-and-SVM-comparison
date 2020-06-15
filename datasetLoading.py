import csv

def loadBreastCancer():
    data = open("datasets/wdbc.data")
    lines = data.readlines()
    dataInputs = []
    dataOutputs = []
    for line in lines:
        splitData = line.split(",")
        currentSample = []
        for i in range(len(splitData)):
            if (i == 1):
                dataOutputs.append(splitData[i])
            elif (i > 1):
                currentSample.append(float(splitData[i]))
        dataInputs.append(currentSample)
        # 0 is M, 1 is B
        for i in range(len(dataOutputs)):
            if (dataOutputs[i] == "M"):
                dataOutputs[i] = 0
            elif (dataOutputs[i] == "B"):
                dataOutputs[i] = 1
    data.close()
    return dataInputs, dataOutputs

def loadCreditCard():
    dataInputs = []
    dataOutputs = []
    with open("datasets/default of credit card clients.csv", "r", newline='') as readCsvfile:
        reader = csv.DictReader(readCsvfile)
        init = True
        counter = 0
        for instance in reader:
            currentSample = []
            if (init):
                init = False
                continue
            for key, value in instance.items():
                if (key == "Y"):
                    dataOutputs.append(int(value))
                elif (key != ""):
                    currentSample.append(int(value))
            dataInputs.append(currentSample.copy())
            counter = counter + 1
            if (counter > 2000):
                break
    return dataInputs, dataOutputs