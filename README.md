# PeakDetective

Software for detecting and curating LC/MS peaks

# Installation

## Install PeakDetective

```
pip install PeakDetective
```

## Install R packages

If you want to run mz.unity to remove degeneracies or run XCMS: the relevant R packages must also be installed (>20 min)
This assumes R (>v4.0) has been installed on the computer and added to the path. 

```
import PeakDetective.detection_helper as detection_helper
detection_helper.PeakList().installRPackages()
```


# Required packages

pyteomics

numpy

scipy

matplotlib

tensorflow

scikit-learn

fastdtw


# Google Colaboratory

PeakDetective can be run as a Google Colaboratory (Colab) notebook, which does not require installation of any software locally. This notebook 
provides many use cases of PeakDetective and corresponding instructions

The Colab notebook can be found [here](https://drive.google.com/file/d/1GIoy8wBrO7KEi2DYm43Auey8uMC2lnzK/view?usp=sharing).


# Usage

Before you start, download example data

```
import os
os.system("git clone https://github.com/pattilab/example_data_for_colab.git")
```

## Load relevant packages

```
import PeakDetective
import PeakDetective.detection_helper as detection_helper
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
import pickle as pkl
import os
import shutil
```

## Define parameters

```
#model parameters, not recomended to change
resolution = 60 #number of data point in 1 min EIC
window = 1.0 #size of EIC window (in minutes)
min_peaks = 100000 #minimum number of peaks used to train autoencoder
smooth_epochs = 10 #number of smoothing epochs to use
batchSizeSmoother = 64 #batch size for training smoother
validationSplitSmoother = 0.1 #fraction of data used for validation split when training smoother
minClassifierTrainEpochs = 200 #minimum number of epochs for training classifier
maxClassifierTrainEpochs = 1000 #maximum number of epochs for training classifier
classifierBatchSize = 4 #batch size for training classifier
randomRestarts = 1 #number of random restarts to use when training classifier
numValPeaks = 10 #number of validaiton peaks for user to label
numPeaksPerRound = 10 #number of peaks to label per active learning iteration
numCores = 10 #number of processor cores to use
numDataPoints = 3 #number of consequetive scans above noise to count as ROI
```

## Training/loading the PeakDetective models

Before the workflows detailed in this notebook can be run, the nueral networks that underpin PeakDetective must be trained or the trained weights must be loaded

### Training

Step 1: load raw data and peak list

What filename is your uploaded file or what we should name your generated peak list? If you need to generate a peak list, see Step 3 of the curating peak list section below for running XCMS

```
peakFile = "example_data_for_colab/xcms_example_data/peaks_formatted.csv" #path to peak file
```

Step 2: Where is your data?

```
mzmlFolder = "example_data_for_colab/xcms_example_data/" #path to raw data folder
```

Step 3: Set data parameters: 

```
ms1ppm = 25.5 #ppm tolerance
align = True  #whether to perform RT alignment or not (True=align, False=do not align)
```

Step 4: Load data:

```
#load peak list
peaklist = pd.read_csv(peakFile)[["mz","rt"]]

#load raw data
raw_data = []
for file in [x for x in os.listdir(mzmlFolder) if ".mzML" in x]:
    temp = PeakDetective.rawData()
    temp.readRawDataFile(mzmlFolder + "/" + file,ms1ppm)
    raw_data.append(temp)
```

Step 5: Initialize PeakDetective object


```
integ = PeakDetective.PeakDetective(numCores = numCores,resolution=resolution)
```

Step 6: Generate EICs

```
if __name__ == "__main__":
    X = integ.makeDataMatrix(raw_data,peaklist["mz"].values,peaklist["rt"].values,align=align)
```

Step 7: Train smoother

```
if __name__ == "__main__":
    integ.trainSmoother(peaklist,raw_data,min_peaks,smooth_epochs,batchSizeSmoother,validationSplitSmoother)
```

Step 8: Train classifier. First you will be prompted to label peaks that will be used as the validation data to see how the model performs. Afterwards, each active learning iteration will prompt additional peaks to be labeled. These are used to train the model. After each iteration a histogram of the scores for all peaks will be plotted and the accuacy of the model on training and validation data will be reported. After 5-10 iterations, a bimodal distribution of scores will be produced and the val_mean_absolute_error will drop dramatically. At this point, the model is trained and the FDR and TPR estimates based on the validation data will be reported for each cutoff. Inspection on this plot is used to set the threshold to use when applying the model.

```
if __name__ == "__main__":
    #train classifier
    integ.trainClassifierActive(X,[],[],minClassifierTrainEpochs,maxClassifierTrainEpochs,classifierBatchSize,randomRestarts,numVal = numValPeaks,numManualPerRound=numPeaksPerRound,inJupyter=True)
```

Based on the TPR and FDR curves plotted above, set the cutoff based on your own preferences for specificity and sensitivity. 

Set the cutoff

```
cutoff = 0.8
```

Step 9: save the model

```
integ.save(mzmlFolder+"PeakDetectiveObject/")
```

### Loading trained model

Step 1: set path to PeakDetective weights

```
weightsPath = "example_data_for_colab/xcms_example_data/PeakDetectiveObject/"
```

Step 2: intitalize PeakDetective object and load weights

```
integ = PeakDetective.PeakDetective(numCores = numCores,resolution=resolution)
integ.load(weightsPath)
```

Set the cutoff

```
cutoff = 0.8
```

## Curating a peak list:

Step 1: Upload or generate a peak list

What filename is your uploaded file or what we should name your generated peak list?

```
peakFile = "example_data_for_colab/xcms_example_data/peaks.csv" #path to peak file
```

Step 2: Where is your data?

```
mzmlFolder = "example_data_for_colab/xcms_example_data/" #path to raw data folder
```

Step 3: Run XCMS to generate peak list (skip if you have already done this)

What parameters do you want to use?

```
ms1ppm = 25.5 #ppm tolerance
peakWidth = (13.8,114.6) #retentin time peak width range (in seconds)
s2n = 13.6 #signal to noise cutoff
noise = 1 #noise threshold
mzDiff = 0.0144 #minimum meaningful m/z difference
prefilter = 5 #number of required consecutive scans to consider for ROI detection
```

Now run XCMS

```
det = detection_helper.PeakList()
det.runXCMS(mzmlFolder, peakFile.split("/")[-1], "negative", ms1ppm, peakWidth,s2n=s2n,noise=noise,mzDiff=mzDiff,prefilter=prefilter)
```

Step 4: Load raw data and peak file

Read in peak list

```
det = detection_helper.PeakList()
det.readXCMSPeakList(mzmlFolder+"peaks.csv")
peaklist = det.peakList[["mz","rt"]]
peaklist.to_csv(mzmlFolder + "peaks_formatted.csv")
```

Read in raw data, enter the ppm tolerance to use in the cell below:

```
ms1ppm = 25.5 #ppm tolerance
```

Load raw data

```
raw_data = []
for file in [x for x in os.listdir(mzmlFolder) if ".mzML" in x]:
    temp = PeakDetective.rawData()
    temp.readRawDataFile(mzmlFolder + "/" + file,ms1ppm)
    raw_data.append(temp)
```

Step 5: Curate peaks, in the cell below set align = True to perform RT alginment of samples

```
align = True #whether to perform RT alignment or not (True=align, False=do not align)
```

curate peaks

```
if __name__ == "__main__":
    peak_curated,peak_scores,peak_intensities = integ.curatePeaks(raw_data,peaklist,threshold=cutoff,align=align)
```

Step 6: filter peaks only present in a particular fraction of samples.

```
detectFrac = 0.5 #minimum fraction of samples where a metabolite must be detected

toDrop = []
sampleCols = [x.filename for x in raw_data]
for index,row in peak_curated.iterrows():
    if np.sum(row[sampleCols])/len(sampleCols) < detectFrac:
        toDrop.append(index)
peak_curated = peak_curated.drop(toDrop,axis=0)
peak_scores = peak_scores.drop(toDrop,axis=0)
peak_intensities = peak_intensities.drop(toDrop,axis=0)
```

Step 7: output results

```
peak_curated.to_csv(mzmlFolder + "peaks_curated.csv")
peak_scores.to_csv(mzmlFolder + "peak_scores.csv")
peak_intensities.to_csv(mzmlFolder + "peak_intensities.csv")
```

## Detecting Peaks


Step 1: where is your data:

```
mzmlFolder = "example_data_for_colab/xcms_example_data/" #path to raw data folder
```

Step 2: set parameters

```
detectFrac = 0.5 #fraction of samples were a feature must be detected
noise = 1000 #noise threshold for finding ROIs (set to baseline noise value)
ms1ppm = 25.5 #ppm tolerance
align = True #whether to perform RT alignment
window = 0.1 #minimum expected spacing between peaks in minutes. Generally 0.05 to 0.2 works well.
```

Step 3: load data

```
#load raw data
raw_data = []
for file in [x for x in os.listdir(mzmlFolder) if ".mzML" in x]:
    temp = PeakDetective.rawData()
    temp.readRawDataFile(mzmlFolder + "/" + file,ms1ppm)
    raw_data.append(temp)
```

Step 4: Detect peaks

```
if __name__ == "__main__":
    peak_scores_pd_det, peak_intensities_pd_det,rois = integ.detectPeaks(raw_data, cutoff=cutoff, intensityCutoff = noise,numDataPoints=numDataPoints,window=window,align=align,detectFrac=detectFrac)
```

Step 5: output results

```
peak_scores_pd_det.to_csv(mzmlFolder + "peak_scores_pd.csv")
peak_intensities_pd_det.to_csv(mzmlFolder + "peak_intensities_pd.csv")
```


## Integrating a peak list


Step 1: Upload or generate a peak list

What filename is your uploaded file or what we should name your generated peak list?

```
peakFile = "example_data_for_colab/xcms_example_data/peaks.csv" #path to peak file
```

Step 2: Where is your data?

```
mzmlFolder = "example_data_for_colab/xcms_example_data/" #path to raw data folder
```

Step 3: set parameters

```
align=True
ms1ppm = 25.5
```

Step 4: load data

```
peaklist = pd.read_csv(peakFile,index_col=0)[["mz","rt"]]

#load raw data
raw_data = []
for file in [x for x in os.listdir(mzmlFolder) if ".mzML" in x]:
    temp = PeakDetective.rawData()
    temp.readRawDataFile(mzmlFolder + "/" + file,ms1ppm)
    raw_data.append(temp)
```

Step 5: integrate peaks

```
if __name__ == "__main__":
    X = integ.makeDataMatrix(raw_data,peaklist["mz"],peaklist["rt"].values,align=align)
    for r in raw_data:
        peaklist[r.filename] = 1.0
    peak_areas = integ.performIntegration(X, [r.filename for r in raw_data], peaklist, cutoff, defaultWidth=0.5,smooth=False)
```

Step 6: output result

```
peak_areas.to_csv(mzmlFolder + "peak_intensities_integration.csv")
```

## Filltering peak list for contaminants, redundancy

Step 1: load peak list

```
peakFile = "example_data_for_colab/xcms_example_data/peak_intensities_integration.csv" #path to peak file
```

```
df = pd.read_csv(peakFile,index_col=0)
```

Step 2: convert to PeakList object

```
det = detection_helper.PeakList()
det.from_df(df)
```

Step 3: impute missing values

```
det.imputeRowMin(det.sampleCols)
```

Step 4: remove contaminants

first we have to set keys that are unique to blank and non-blank samples. The name of the peak area columns are listed by running the cells below. Enter the keys unique to blanks and non-blanks in the cell following. Additionally, enter the factor that sets how many times higher the mean intensity in the non-blank samples must be in comparision to the blanks.

```
blankKey = ["blank"]
sampleKey = ["NIST"]
factor = 2
```

Now run background subtraction

```
det.backgroundSubtract(blankKey,sampleKey,factor)
```

Step 5: remove redundancies (isotopes, fragments, adducts, etc.) 

First we have to set some thresholds for the correlation between redundant features and the retention time difference between redundant features

```
corrThresh = 0.9 #minimum correlation of intensities in sample columns to consider as redundant
rtThresh = 0.5 #maximum difference in RT to consider as redundant (in minutes)
polarity = -1 #polarity of data (-1 for negative mode, 1 for postive mode
ms1ppm = 25.5 #ppm tolerance
```

remove redundancies

```
if __name__ == "__main__":
    det.removeRedundancy(corrThresh,rtThresh,polarity,ms1ppm,numCores,sampleCols=det.sampleCols)
```

Step 6 output results

```
det.to_csv(peakFile.replace(".csv","_refined.csv"))
```
