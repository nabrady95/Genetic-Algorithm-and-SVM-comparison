import random
import time
import numpy
from GA import geneticAlgorithm
import warnings
from datasetLoading import loadBreastCancer, loadCreditCard
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")

def initialisePopulation(populationSize):
    population = []
    for i in range(populationSize):
        parameters = {}
        parameters["populationSize"] = int(random.randint(6,10))
        parameters["crossoverRate"] = random.uniform(0.25,0.75)
        while (int(parameters["crossoverRate"]*parameters["populationSize"]) < 2):
                parameters["crossoverRate"] = parameters["crossoverRate"] + 0.01
        parameters["mutationRate"] = random.uniform(0.05,0.15)
        population.append(parameters)
    return population
    
def calcTerminalTimeCondition(currentTime, startTime, runTime):
    print(str(runTime - (currentTime - startTime)) + " seconds left")
    return ((currentTime - startTime) > runTime)

def crossover(parameterSet1, parameterSet2):
    # single point midway crossover
    crossoverPoint = numpy.uint32(len(parameterSet1)/2)
    newParameterSet = {}
    counter = 0
    for key,  value in parameterSet1.items():
        if (counter < crossoverPoint):
            newParameterSet[key] = value
        else:
            newParameterSet[key] = parameterSet2[key]
        counter = counter + 1
    return newParameterSet

def mutate(parameterSet, mutationChance):
    for key, value in parameterSet.items():
        if (key == "populationSize"):
            if (random.uniform(0,1) <= mutationChance):
                popSizeMutations = [-1,1]
                parameterSet[key] = parameterSet[key] + random.choice(popSizeMutations)
            if (populationSize < 4):
                parameterSet[key] = random.randint(4,6)
        elif (key == "crossoverRate"):
            if (random.uniform(0,1) <= mutationChance):
                parameterSet[key] = parameterSet[key] + random.uniform(-0.1, 0.1)
            while (int(parameterSet["crossoverRate"]*parameterSet["populationSize"]) < 2):
                parameterSet["crossoverRate"] = parameterSet["crossoverRate"] + 0.01
        elif (key == "mutationRate"):
            parameterSet[key] = parameterSet[key] + random.uniform(-0.02, 0.02) 
    return parameterSet

def evolvePopulation(population, fitness, crossoverRate, mutationChance):
    parentsForBreeding = []
    populationSize = len(population)
    
    for parent in range(int(crossoverRate*len(population))):
        parentId = numpy.where(fitness == numpy.max(fitness))
        parentsForBreeding.append(population[parentId[0][0]])
        population.remove(population[parentId[0][0]])
        fitness.remove(fitness[parentId[0][0]])
        
    newGeneration = parentsForBreeding.copy()
    for offspring in range(populationSize - len(newGeneration)):
        parentIds = random.sample(range(0,len(parentsForBreeding)), 2)
        newGeneration.append(mutate(crossover(parentsForBreeding[parentIds[0]], parentsForBreeding[parentIds[1]]), mutationChance))
    return newGeneration

# Load data
dataInputs, dataOutputs = loadCreditCard()
dataInputs = numpy.array(dataInputs)
dataOutputs = numpy.array(dataOutputs)
noOfClasses = len(numpy.unique(dataOutputs))
neuronPerLayer = int(len(dataInputs[0])/3)

# Set hyperparameters
populationSize = 6
crossoverRate = 0.5
mutationChance = 0.75
runTime = 7200
runs = 3

# Algorithm start
population = initialisePopulation(populationSize)
terminalCondition = False
startTime = time.time()
generation = 0
topFitness = []

# Used for fitness
std = StandardScaler()
kf = KFold(n_splits=5)
kf.get_n_splits(dataInputs)

while (not terminalCondition):
    print("Generation " + str(generation))
    for network in population:
        print(network)
    fitness = []
    paramNo = 0
    for paramset in population:
        runResults = []
        print("Testing paramset " + str(paramNo))
        for run in range(runs):
            crossFoldResults = []
            for train_index , test_index in kf.split(dataInputs):
                X_train, X_test= dataInputs[train_index], dataInputs[test_index]
                y_train, y_test= dataOutputs[train_index], dataOutputs[test_index]
                #Transform data 
                X_train = std.fit_transform(X_train)
                X_test = std.transform(X_test)
                acc, network = geneticAlgorithm(paramset["populationSize"], paramset["crossoverRate"], paramset["mutationRate"], [neuronPerLayer,neuronPerLayer,noOfClasses], 10, X_train, y_train, False, False, False, 0)
                accuracyTest=network.evaluate(X_test, y_test)
                crossFoldResults.append(accuracyTest)
            runResults.append(numpy.mean(crossFoldResults))
        paramNo = paramNo + 1
        fitness.append(numpy.median(runResults))
    topFitness.append(max(fitness))
    print(topFitness)
    print("Total time: " + str(time.time() - startTime))
    population = evolvePopulation(population, fitness, crossoverRate, mutationChance)
    terminalCondition = calcTerminalTimeCondition(time.time(), startTime, runTime)
    generation = generation + 1

print("======================================")
print("Final population")
for network in population:
    print(network)