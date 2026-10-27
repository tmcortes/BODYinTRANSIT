import torch
import random

from dataset import SubjectDataset, Subject, Condition, Walk

def saveCompleteDataset():
    
    folderPath = '../dataset/unified_db' 
    excludeIds = ['p021', 'p028', 'p051', 'p288']
    #excludeFeat = (11, 307, 310)

    winLength = 150
    featuresIdx = list(range(0,306))
    #featuresIdx = [num for num in featuresIdx if num not in excludeFeat]
    splitPcts= {'train': 100, 'val': 0, 'test': 0}
    seed=1

    datasetTr = SubjectDataset(folderPath, excludeIds, mode='train', 
                 winLength = winLength, featuresIdx = featuresIdx,
                 splitPcts= splitPcts, seed=42)
    
    torch.save({'dataset': datasetTr}, 'train_dataset_'  +str(winLength)  + '.pth')
    
    return datasetTr


def getRandomSample( dataset, subjectId ):

    # pick a window randomly from the dataset
    idx = dataset.subjectIds.index(subjectId)
    c = random.randint(0,2)
    w = random.randint(0, len(dataset.subjectObj[idx].conditions[c].walks)-1)
    walk = dataset.subjectObj[idx].conditions[c].walks[w]

    X = walk.sample(dataset.winLength)
    return X

def getRangeMask(X, p05, p95):

    chunkRange = X.max(dim=0).values - X.min(dim=0).values
    inRange = (chunkRange >= p05) & (chunkRange <= p95)
    return inRange  # Tensor booleano shape (306,)


def computeStats(train_dataset, numWindows=1000):

    ranges = torch.zeros([numWindows, 306])
    goodIds = ['p109', 'p173', 'p176', 'p195', 'p320']

    for i in range(numWindows):

        X = getRandomSample(train_dataset, random.choice(goodIds))
        ranges[i] = X.max(dim=0).values - X.min(dim=0).values
    
    p05 = torch.quantile(ranges, 0.05, dim=0)
    p95 = torch.quantile(ranges, 0.95, dim=0)

    quantilesDict = {
    'p05': p05,
    'p95': p95
    }

    torch.save(quantilesDict, 'sensorRangeQuantiles.pth')
    return quantilesDict

def estimateZeroedSensors(trainDataset, quantilesDict, numSamples=10000):
    p05 = quantilesDict['p05']
    p95 = quantilesDict['p95']

    totalZeroFrac = 0.0

    for _ in range(numSamples):
        subjectId = random.choice(trainDataset.subjectIds)
        X = getRandomSample(trainDataset, subjectId)
        mask = getRangeMask(X, p05, p95)  # True = dentro de rango
        fracZero = (~mask).float().mean().item()  # proporción de sensores fuera de rango
        totalZeroFrac += fracZero

    avgZeroFrac = totalZeroFrac / numSamples

    print(f'Average columns to zero on each input X: {avgZeroFrac*100:.2f}%')
    return avgZeroFrac



if __name__ == '__main__':

    # Load the dataset, create it first if neccessary
    #train_dataset = saveCompleteDataset()
    train_dataset = torch.load('train_dataset_150.pth')['dataset']

    # Let's compute the stats from the subjects we know are better by visual inspection
    #quantilesDict = computeStats(train_dataset, numWindows=100000)

    loadedQuantiles = torch.load('sensorRangeQuantiles.pth')

    quantilesDict = {
        'p05': loadedQuantiles['p05'],
        'p95': loadedQuantiles['p95']
    }

    estimateZeroedSensors(train_dataset, quantilesDict, numSamples=10000)