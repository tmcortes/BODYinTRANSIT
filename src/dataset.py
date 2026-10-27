import os
import re
import random
from sklearn.model_selection import train_test_split
from natsort import natsorted
import pandas as pd
import glob
import statistics

import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

def extract_number(s: str) -> int:
    match = re.match(r"p0*(\d+)", s)
    if match:
        return int(match.group(1))
    else:
        raise ValueError(f"String {s} not in expected format")
    
def downsample_labels_tensor(labels: torch.Tensor, factor: int = 3) -> torch.Tensor:
    T = labels.shape[0]
    trim_start = T % factor
    labels_trimmed = labels[trim_start:]

    reshaped = labels_trimmed.view(-1, factor)
    non_zero_mask = reshaped != 0
    first_non_zero_idx = non_zero_mask.float().argmax(dim=1)
    has_non_zero = non_zero_mask.any(dim=1)

    batch_indices = torch.arange(reshaped.size(0), device=labels.device)
    downsampled = reshaped[batch_indices, first_non_zero_idx]
    downsampled[~has_non_zero] = 0

    # Validation check: ensure label sum matches
    original_sum = labels.sum().item()
    downsampled_sum = downsampled.sum().item()

    if original_sum != downsampled_sum:
        print(f"⚠️ Label sums do not match! Original: {original_sum}, Downsampled: {downsampled_sum}")

    return downsampled.unsqueeze(1)


class Downsample(torch.nn.Module):
    r"""Downsampling layer that applies anti-aliasing filters.
    For example, order=0 corresponds to a box filter (or average downsampling
    -- this is the same as AvgPool in Pytorch), order=1 to a triangle filter
    (or linear downsampling), order=2 to cubic downsampling, and so on.
    See https://richzhang.github.io/antialiased-cnns/ for more details.
    """

    def __init__(self, channels=None, factor=2, order=1):
        super(Downsample, self).__init__()
        assert factor > 1, "Downsampling factor must be > 1"
        self.stride = factor
        self.channels = channels
        self.order = order

        # Figure out padding and check params make sense
        # The padding is given by order*(factor-1)/2
        # so order*(factor-1) must be divisible by 2
        total_padding = order * (factor - 1)
        assert total_padding % 2 == 0, (
            "Misspecified downsampling parameters."
            "Downsampling factor and order must be such "
            "that order*(factor-1) is divisible by 2"
        )
        self.padding = int(order * (factor - 1) / 2)

        box_kernel = np.ones(factor)
        kernel = np.ones(factor)
        for _ in range(order):
            kernel = np.convolve(kernel, box_kernel)
        kernel /= np.sum(kernel)
        kernel = torch.Tensor(kernel)
        self.register_buffer(
            "kernel", kernel[None, None, :].repeat((channels, 1, 1))
        )

    def forward(self, x):
        return F.conv1d(
            x,
            self.kernel,
            stride=self.stride,
            padding=self.padding,
            groups=x.shape[1],
        )
    
def downsample_signal(x, factor=3, order=1):

    # x comes as [Time, channels] so we have to change it to [batch, channels, time]
    x = x.T.unsqueeze(0)  # [1, C, T]
    downsampler = Downsample(channels=x.shape[1], factor=factor, order=order)
    x_ds = downsampler(x)
    return x_ds.squeeze(0).T  # back to [T', C]

def sigmoidNormalize(X):
    Xnorm = X.clone()
    q75 = torch.quantile(Xnorm, 0.75, dim=0)
    q25 = torch.quantile(Xnorm, 0.25, dim=0)
    trainScale = (q75 - q25)

    Xnorm.div_(trainScale + 1e-3)
    torch.asinh_(Xnorm)
    return Xnorm

def _sigmoidNormalize(X):

    q75 = torch.quantile(X, 0.75, dim=0)
    q25 = torch.quantile(X, 0.25, dim=0)
    trainScale = (q75 - q25)

    X.div_(trainScale + 1e-3)
    torch.asinh_(X)
    return X

class Walk():

    def __init__(self, data, startRow, endRow, label, source):

        self.data = data
        self.startRow = startRow
        self.endRow = endRow
        self.label = label
        self.originalLabel = label
        self.source = source
        return
    
    def sample(self, winLength):
        
        s = random.randint(self.startRow, self.endRow-winLength)
        e = s + winLength
        return self.data[s:e]
    
    def __str__(self):

        print("\tData shape: (%d,%d), Start-end: (%d,%d), Label: %d" %(
            self.data.shape[0], self.data.shape[1], self.startRow, self.endRow, self.label))
        return ""

class Condition():

    def __init__(self, SubjectId, folderPath, conditionName, featuresIdx, winLength):

        self.id = SubjectId
        self.folderPath = folderPath
        self.condition = conditionName
        self.featuresIdx = featuresIdx
        self.winLength = winLength

        # these attributes will be filled by the following functions
        self.rb1Data = None
        self.rb2Data = None
        self.labels1 = None
        self.labels2 = None

        # each condition has 12 walks objects
        self.walks = []

        # load the walks
        self._loadCSVs()
        self._setWalks()
        return
    
    def _setWalks(self):

        # find the labels of each walk during the condition
        int1 = self._findIntervals(self.labels1)
        int2 = self._findIntervals(self.labels2)

        for i,(s,e,l) in enumerate(int1):

            # we skip the first one (walking static over the same place)
            if i == 0: continue

            # we skip a walk if its len is less than winLength
            if e-s <= self.winLength: continue

            self.walks.append(Walk(data=self.rb1Data, startRow=s, endRow=e, label=l, source='rokoko1'))

        for i,(s,e,l) in enumerate(int2):
            if i == 0: continue
            if e-s <= self.winLength: continue
            self.walks.append(Walk(data=self.rb2Data, startRow=s, endRow=e, label=l, source='rokoko2'))
        
        return
    
    def _findIntervals(self, tensor):
        intervals = []
        start = 0
        label = 0

        for i in range(len(tensor)):
            
            if tensor[i] != 0:  # Found a new nonzero value

                intervals.append((start, i, int(tensor[i].item())))
                start = i+1


        return intervals

    def _readCSV(self, path):

        df = pd.read_csv(path, header=None, low_memory=False)
        df = df.iloc[1:, 1:]

        data = torch.tensor(df.to_numpy(dtype=float), dtype=torch.float32)
        return data
    
    def _loadCSVs(self):

        paths = self._createCSVPaths()

        # exceptions:

        self.rokoko1Data = self._readCSV(paths['rokoko1Path'])
        self.rokoko2Data = self._readCSV(paths['rokoko2Path'])
        #self.bitalino1Data = self._readCSV(paths['bitalino1Path'])
        #self.bitalino2Data = self._readCSV(paths['bitalino2Path'])
        self.labels1 = self._readCSV(paths['labels1Path'])
        self.labels2 = self._readCSV(paths['labels2Path'])

        # #usually, there are few miliseconds more samples in bitalino than rokoko, 
        # #we just trimm those
        # n11 = self.rokoko1Data.shape[0]
        # n12 = self.bitalino1Data.shape[0]
        # n21 = self.rokoko2Data.shape[0]
        # n22 = self.bitalino2Data.shape[0]

        # if n11 > n12:
        #     self.rokoko1Data = self.rokoko1Data[0:n12]
        #     self.labels1 = self.labels1[0:n12]

        # if n11 < n12:
        #     self.bitalino1Data = self.bitalino1Data[0:n11]

        # if n21 > n22:
        #     self.rokoko2Data = self.rokoko2Data[0:n22]
        #     self.labels2 = self.labels2[0:n22]

        # if n21 < n22:
        #     self.bitalino2Data = self.bitalino2Data[0:n21]
        
        #rb1Data = torch.cat((self.rokoko1Data, self.bitalino1Data), dim=1)
        #rb2Data = torch.cat((self.rokoko2Data, self.bitalino2Data), dim=1)

        rb1Data = self.rokoko1Data
        rb2Data = self.rokoko2Data

        self.rb1Data = rb1Data[:, self.featuresIdx]
        self.rb2Data = rb2Data[:, self.featuresIdx]

        #self.rb1Data = _sigmoidNormalize(self.rb1Data)
        #self.rb2Data = _sigmoidNormalize(self.rb2Data)

        # let's change the sample rate from 100hz to 33hz because we are going to use
        # the HarNet from: https://github.com/OxWearables/ssl-wearables/tree/main
        self.rb1Data = downsample_signal(self.rb1Data)
        self.rb2Data = downsample_signal(self.rb2Data)

        # we also have to "downsample" the labels (this is done carefully in a function)
        self.labels1 = downsample_labels_tensor(self.labels1)
        self.labels2 = downsample_labels_tensor(self.labels2)

        if torch.any(torch.isnan(self.rb1Data)) or torch.any(torch.isnan(self.rb2Data)):
            print(self.id, self.condition)
            nan_mask = torch.isnan(rb1Data)
            columns_with_nans = torch.any(nan_mask, dim=0)
            nan_column_indices = torch.nonzero(columns_with_nans).squeeze().tolist()
            print("Columns with NaNs:", nan_column_indices)
            a = 3

        assert(self.rb1Data.shape[0] == self.labels1.shape[0])
        assert(self.rb2Data.shape[0] == self.labels2.shape[0])
        return
    
    def _createCSVPaths(self):

        # create rokoko file paths
        r1d = (self.id + '-' + self.condition + '-1_rokoko_data_unified.csv').lower()
        r1l = (self.id + '-' + self.condition + '-1_rokoko_label_unified.csv').lower()
        r2d = (self.id + '-' + self.condition + '-2_rokoko_data_unified.csv').lower()
        r2l = (self.id + '-' + self.condition + '-2_rokoko_label_unified.csv').lower()

        r1d_path = os.path.join(self.folderPath, self.id, 'Rokoko', 'CSV', r1d )
        r1l_path = os.path.join(self.folderPath, self.id, 'Rokoko', 'CSV',r1l )
        r2d_path = os.path.join(self.folderPath, self.id, 'Rokoko', 'CSV',r2d )
        r2l_path = os.path.join(self.folderPath, self.id, 'Rokoko', 'CSV',r2l )

        # create bitalino file paths
        b1d = (self.id + '-' + self.condition + '-1_data_unified.csv').lower()
        b1l = (self.id + '-' + self.condition + '-1_label_unified.csv').lower()
        b2d = (self.id + '-' + self.condition + '-2_data_unified.csv').lower()
        b2l = (self.id + '-' + self.condition + '-2_label_unified.csv').lower()

        b1d_path = os.path.join(self.folderPath, self.id, 'BITalino', 'CSV',b1d )
        b1l_path = os.path.join(self.folderPath, self.id, 'BITalino', 'CSV',b1l )
        b2d_path = os.path.join(self.folderPath, self.id, 'BITalino', 'CSV',b2d )
        b2l_path = os.path.join(self.folderPath, self.id, 'BITalino', 'CSV',b2l )
        
        paths = {'rokoko1Path': r1d_path, 'rokoko2Path': r2d_path,
                 'bitalino1Path': b1d_path, 'bitalino2Path': b2d_path,
                 'labels1Path': r1l_path, 'labels2Path': r2l_path}
                 
        
        return paths
    
    def __str__(self):

        print("Condition: %s" %(self.condition))
        for i,w in enumerate(self.walks):
            print("\tWalk %d" %(i))
            print(w)

        return ""
    
class Subject():

    def __init__(self, id, folderPath, featuresIdx, winLength, label):

        self.id = id
        self.folderPath = folderPath
        self.featuresIdx = featuresIdx
        self.label = label

        self.controlCondition = Condition(id, folderPath, 'Control', featuresIdx, winLength)
        self.highCondition = Condition(id, folderPath, 'High', featuresIdx, winLength)
        self.lowCondition = Condition(id, folderPath, 'Low', featuresIdx, winLength)
        self.conditions = [self.controlCondition, self.highCondition, self.lowCondition]

        self._relabelConditions()

        return

    def _relabelConditions(self):

        if self.label == "condition":
            # We use as labels the audio conditions:
            # 0: low freq audio, 1: control, 2: high freq audio
            for walk in self.lowCondition.walks:
                walk.label = 0

            for walk in self.controlCondition.walks: 
                walk.label = 1

            for walk in self.highCondition.walks:
                walk.label = 2

            return
             

        if self.label == "happiness" or self.label == "weight":

            # Use the complete trial-label table.
            df = pd.read_excel("../dataset/allLabels/labels.xlsx")

            # We use the new relative labeling scheme: -1, 0, +1
            id = extract_number(self.id)

            # 1. Calculate the baseline for this subject
            # Get the two scores from the control trials and calculate their mean.
            control_scores = df.loc[(df["id"] == id) & (df["condition"] == 'control'), self.label]
            baseline = control_scores.mean()

            # 2. Label the control trials as 0
            # All walks from the control condition are the reference, so their label is 0.
            for walk in self.controlCondition.walks:
                walk.label = 0 + 1

            # 3. Label the high and low frequency trials relatively
            # We iterate through the other two conditions to assign +1 or -1.
            for condition_obj, condition_name in [(self.lowCondition, 'low'), (self.highCondition, 'high')]:

                for walk in condition_obj.walks:

                    # Get the original score for this walk
                    repetition = 1 if walk.source == 'rokoko1' else 2
                    original_score = df.loc[(df["id"] == id) & (df["condition"] == condition_name) & (df["repetition"] == repetition), self.label].iloc[0]

                    # Compare with the baseline to assign the new label
                    if original_score > baseline:
                        walk.label = 1 + 1
                    elif original_score < baseline:
                        walk.label = -1 + 1
                    else:
                        walk.label = 0 + 1
            
            return

        return
    
    def __getitem__(self, index):
        return self.conditions[index]
    
    def __str__(self):

        print("SubjectId: %s" %(self.id))
        print("Folder: %s" %(self.folderPath))

        print(self[0])
        print(self[1])
        print(self[2])
        return ""


class WalksDataset(Dataset):

    def __init__(self, walks, winLength, featuresIdx):

        self.walks = walks
        self.winLength = winLength

        # Load stats for filtering
        loadedQuantiles = torch.load('sensorRangeQuantiles.pth')
        self.p05 = loadedQuantiles['p05'][featuresIdx]
        self.p95 = loadedQuantiles['p95'][featuresIdx]

        return
    
    def scale(self, X):

        # X: [T, C]
        X_min = X.min(dim=0, keepdim=True).values  # [1, C]
        X_max = X.max(dim=0, keepdim=True).values  # [1, C]

        scale = X_max - X_min
        scale[scale == 0] = 1.0  # avoid zero-division

        X_scaled = 2 * (X - X_min) / scale - 1
        return X_scaled
    
    
    def maskFeatures(self, X):
        chunkRange = X.max(dim=0).values - X.min(dim=0).values
        mask = (chunkRange >= self.p05) & (chunkRange <= self.p95)  # booleano (312,)

        Xmasked = X.clone()
        # null colums outside typicall range (either noise peaks, or muted-failing sensors)
        Xmasked[:, ~mask] = 0

        return Xmasked
     
    def __getitem__(self, idx):

        # pick a walk randomly from the dataset

        walk = random.choice(self.walks)
        X = walk.sample(self.winLength)
        y = walk.label

        # use the percentile stats (self.p05, self.p95) to mask bad sensor-axis
        X = self.maskFeatures(X)

        # scale the data linearly to the (-1) -> (+1) range
        X = self.scale(X)

        #prepare data dims and type
        y = torch.tensor(y, dtype=torch.long)

        return X,y
    
    def __len__(self):

        if not self.walks:
            return 0

        samplesPerWalk = int(self.walks[0].data.shape[0] / self.winLength)
        return len(self.walks) * samplesPerWalk * 10


class DatasetManager():

    def __init__(self, folderPath, id_, winLength = 300, 
                 featuresIdx = list(range(0,312)),
                 splitPcts= {'train': 60, 'val': 20, 'test': 20}, label=None, seed=42):
        
        self.folderPath = folderPath
        self.winLength = winLength
        self.featuresIdx = featuresIdx
        self.splitPcts = splitPcts
        self.label = label
        self.seed = seed
        self.patientId = id_
        self.patientObj = []
        self.numSamples = -1

        self.trWalks = []
        self.valWalks = []
        self.testWalks = []

        # 1 - Load the patient
        self.loadPatient()

        # 2 - split data in train, val, test
        self.splitData()

    def loadPatient(self):

        print("Loading patient (%s)" %(self.patientId))
        self.patientObj =  Subject(self.patientId, self.folderPath, self.featuresIdx, self.winLength, self.label)
        
        return
    

    def splitData(self):

        random.seed(self.seed)
        
        patient = self.patientObj

        # --- 1. Pool all walks and create stratification keys ---
        all_patient_walks = []
        stratify_keys = []
        for condition in patient.conditions:
            for walk in condition.walks:
                all_patient_walks.append(walk)
                stratify_keys.append(f"{condition.condition}_{walk.label}")

        if len(all_patient_walks) < 2:
            self.trWalks, self.valWalks, self.testWalks = [], [], []
            return

        # Get size percentages
        train_pct = self.splitPcts['train'] / 100.0
        val_pct = self.splitPcts['val'] / 100.0
        test_pct = self.splitPcts['test'] / 100.0
        
        # Size of the temporary set (val + test) for the first split
        temp_size_pct = val_pct + test_pct
        # Size of the test set relative to the temporary set for the second split
        # Add a small epsilon to avoid division by zero if temp_size_pct is 0
        test_size_rel_pct = test_pct / (temp_size_pct + 1e-9)

        # --- 2. Check if stratification is possible and split ---
        unique_keys, counts = np.unique(stratify_keys, return_counts=True)
        if np.min(counts) < 2:
            # Fallback to simple random split
            print(f"Warning: Stratification not possible for patient {patient.id} seed {self.seed}. Using random split.")
            
            train_walks, temp_walks = train_test_split(all_patient_walks, test_size=temp_size_pct, random_state=self.seed)
            val_walks, test_walks = train_test_split(temp_walks, test_size=test_size_rel_pct, random_state=self.seed)
        else:
            # Proceed with the robust two-phase stratified split
            train_walks, temp_walks, _, temp_keys = train_test_split(
                all_patient_walks, stratify_keys, test_size=temp_size_pct, random_state=self.seed, stratify=stratify_keys)
            
            unique_temp_keys, temp_counts = np.unique(temp_keys, return_counts=True)
            if len(temp_walks) < 2 or np.min(temp_counts) < 2:
                val_walks, test_walks = train_test_split(temp_walks, test_size=test_size_rel_pct, random_state=self.seed)
            else:
                val_walks, test_walks, _, _ = train_test_split(
                    temp_walks, temp_keys, test_size=test_size_rel_pct, random_state=self.seed, stratify=temp_keys)

        # --- 3. Directly assign the final flat lists ---
        self.trWalks = train_walks
        self.valWalks = val_walks
        self.testWalks = test_walks
    
        return
    
    
    def createDataset(self, mode):
        
        if mode == 'train':
            return WalksDataset(self.trWalks, self.winLength, self.featuresIdx)
        
        elif mode == 'val':
            return WalksDataset(self.valWalks, self.winLength, self.featuresIdx)
        
        else:
            return WalksDataset(self.testWalks, self.winLength, self.featuresIdx)
    


if __name__ == '__main__':


    # "Dataset "
    folderPath = '../dataset/unified_db'
    label = "weight" 
    id_ = 'p295'

    winLength = 150 # at 30hz is 5 seconds

    # csv columns from the 17 accelerometers in rokoko
    with open('accelerometer_columns.txt', 'r') as f:
        featuresIdx = [int(line.strip()) for line in f]

    splitPcts= {'train': 60, 'val': 20, 'test': 20}
    seed=5

    dm = DatasetManager(folderPath, id_, winLength=150, featuresIdx=featuresIdx, splitPcts=splitPcts, label= label, seed=seed)
    datasetTr = dm.createDataset(mode='train')
    datasetVal = dm.createDataset(mode='val')
    datasetTest = dm.createDataset(mode='test')


