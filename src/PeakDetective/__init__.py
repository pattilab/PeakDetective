from pyteomics import mzml
import sys
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Manager,Pool
from threading import Thread
from tensorflow import keras
from keras.constraints import max_norm
import keras.layers as layers
import scipy.stats as stats
from bisect import bisect_left
import math
import random as rd
import IPython.display
from copy import deepcopy
import pandas as pd
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from scipy.interpolate import interp1d
import pickle as pkl
import os
import sklearn.metrics as met
import datetime
from scipy.signal import find_peaks,peak_widths
import scipy
from sklearn.mixture import GaussianMixture

class PeakDetective():
    """
    Class for curation/detection of LC/MS peaks in untargeted metabolomics data
    """
    def __init__(self,resolution=60,numCores=1,windowSize = 1.0):
        """
        Constructor for PeakDetective object
        @param resolution: int, number of datapoints to sample in the EIC window
        @param numCores: int, number of processor cores to use for concurrent computations
        @param windowSize: float, size of EIC window to extract (in minutes)
        """
        self.resolution = resolution
        self.numCores = numCores
        self.windowSize = windowSize
        self.smoother = Smoother(resolution)
        self.classifier = ClassifierLatent(5)
        self.encoder = keras.Model(self.smoother.input, self.smoother.layers[7].output)

    def save(self,path):
        """
        Save model weights to given path
        @param path: str, path where weights should be saved
        @return: None
        """
        if not os.path.exists(path): os.mkdir(path)
        pkl.dump([self.resolution,self.windowSize],open(path+"parameters.pd","wb"))
        self.smoother.save(path + "smoother.h5")
        self.classifier.save(path+"classifier.h5")
        self.encoder.save(path+"encoder.h5")


    def load(self,path):
        """
        Load model weights
        @param path: str, path to folder where model weights were saved
        @return: None
        """
        try:
            [self.resouton,self.windowSize] = pkl.load(open(path + "parameters.pd","rb"))
            self.smoother = keras.models.load_model(path+"smoother.h5")
            self.classifier = keras.models.load_model(path+"classifier.h5")
            self.encoder = keras.models.load_model(path+"encoder.h5")

        except:
            print("Error: pd and h5 files were not found, check path")

    def plot_overlayedEIC(self,rawdatas,mz,rt_start,rt_end,alpha=0.3):
        """
        Plot an overlayed EIC for specified samples and mz and retention time range
        @param rawdatas: list of rawData objects, samples to plot EIC for
        @param mz: float, m/z value to extract EIC
        @param rt_start: float, retention time value that is the starting value to plot
        @param rt_end: float, retention time value that is the ending value to plot
        @param alpha: float, transparency factor (0-1)
        @return: None
        """
        ts = np.linspace(rt_start,rt_end,self.resolution)
        for data in rawdatas:
            s = data.interpolate_data(mz,rt_start,rt_end)
            ints  = [np.max([x,0]) for x in s(ts)]
            plt.plot(ts,ints,alpha=alpha)


    @staticmethod
    def getNormalizedIntensityVector(data,mzs,rtstarts,rtends,resolution,q=None):
        """
        Get a list of EIC vectors from a single sample
        @param data: rawData, rawData object to extract EICs for
        @param mzs: list, list of m/z values to get EICs for
        @param rtstarts: list, of retention times values that are the lower bound of the EIC window
        @param rtends: list, of retention times values that are the upper bound of the EIC window
        @param resolution: int, number of data points to sample in the EIC window
        @param q: Queue or None, only used for multiprocessing
        @return: numpy array, array of EICs (one row per EIC)
        """
        out = np.zeros((len(mzs),resolution))
        i=0
        for mz,rt_start,rt_end in zip(mzs,rtstarts,rtends):
            s = data.interpolate_data(mz,rt_start,rt_end)
            out[i,:] = s(np.linspace(rt_start,rt_end,resolution))
            i += 1
        if type(q) != type(None):
            q.put(0)
        return out

    def makeDataMatrix(self,rawdatas,mzs,rts,align=False,groups=None):
        """
        make a matrix of EICs that is composed of every EIC for each feature in each file
        @param rawdatas: list of rawData objects, samples to generate EICs from
        @param mzs: list, list of m/z values to get EICs for
        @param rts: list,  list of retention time values to get EICs for
        @param align: bool, whether to perform retention time alignment or not (True = align, False = do not align)
        @param group: list, group key to indicate if EICs should be aligned together
        @return: numpy matrix, matrix of EICs. The order of the matrix is first by samples then by features. For example, for an ouput matrix generated from n samples and m features, would have length n * m and the EIC for the ith feature for the jth sample would be in row: i + (j * m)
        """
        #gather start and end times of EIC windows
        rtstarts = [rt - self.windowSize/2 for rt in rts]
        rtends = [rt + self.windowSize/2 for rt in rts]

        if groups is None:
            groups = list(range(len(mzs)))

        #gather arguments for function
        args = []
        featInds = []
        sampInds = []
        i = 0
        for rawdata in rawdatas:
            featInds += list(groups)
            sampInds += [i for _ in groups]
            args.append([rawdata, mzs, rtstarts, rtends,self.resolution])

        #get EICs
        result = startConcurrentTask(PeakDetective.getNormalizedIntensityVector, args, self.numCores, "forming matrix", len(args))

        #concatenate EICs
        result = np.concatenate(result,axis=0)

        #align EICs
        if align:
            result = alignDataMatrix(result,featInds,sampInds,True,self.numCores)

        return result

    def trainSmoother(self,peaks,raw_datas,numPeaks,smooth_epochs,batch_size,validation_split):
        """
        Train smoothing autoencoder network
        @param peaks: Pandas DataFrame, list of features detected from some peak detection software
        @param raw_datas: list of rawData objects, samples to use for training
        @param numPeaks: int, number of peaks to sample to train autoencoder
        @param smooth_epochs: int, number of epochs to train for
        @param batch_size: int, batch size for training
        @param validation_split: float, fraction of EICs to hold out for validation, between 0-1.
        @return: None
        """
        #generate data matrix
        print("generating EICs...")
        mzs = rd.choices(list(peaks["mz"].values),k=int(numPeaks/len(raw_datas)))
        rts = rd.choices(list(peaks["rt"].values),k=int(numPeaks/len(raw_datas)))

        X = self.makeDataMatrix(raw_datas,mzs,rts)

        #normalize matrix
        X_norm = normalizeMatrix(X)

        print("done")

        #fit autoencoder
        print("fitting smoother...")
        smoother = Smoother(self.resolution)
        smoother.fit(X_norm, X_norm, epochs=smooth_epochs, batch_size=batch_size, validation_split=validation_split,verbose=1)

        #set updated models
        self.smoother = smoother
        self.encoder = keras.Model(smoother.input, smoother.layers[7].output)
        print("done")

    def trainClassifier(self,X,y,X_val,y_val,min_epochs,max_epochs,batch_size,reinitialize=True):
        """
        Train classifer network
        @param X: numpy matrix, input EICs to train on
        @param y: numpy matrix, input labels to train on
        @param X_val: numpy matrix, EICs to use a validation data
        @param y_val: numpy matrix, input labels to use as validation
        @param min_epochs: int, minimum number of epochs to train
        @param max_epochs: int, maximum number of epochs to train
        @param batch_size: int, batch size to use when training
        @param restarts: int, number of random restarts to run
        @return: EarlyStopping callback, training history in keras callback object
        """

        #normalize matrix
        X_norm = normalizeMatrix(X)

        #calculate peak areas
        tics = np.log10(np.array([np.max([2, integratePeak(x)]) for x in X]))

        #get latent rep. of EICs
        X_latent = self.encoder.predict(X_norm)

        #normalize validation EICs
        X_tmp = normalizeMatrix(X_val)

        #get peak areas for validation data
        tic_val = np.log10(np.array([np.max([2, integratePeak(x)]) for x in X_val]))

        #get latent rep. of validaiton EICs
        X_val = self.encoder.predict(X_tmp)

        #make callback object
        cb = keras.callbacks.EarlyStopping(
            monitor="val_loss",
            min_delta=0,
            patience=3,
            verbose=0,
            mode="auto",
            baseline=None,
            restore_best_weights=True,
        )

        history = keras.callbacks.History()

        # initialize classifier
        if reinitialize: classifer = ClassifierLatent(X_latent.shape[1])
        else: classifer = self.classifier

        #train to minimum epochs
        if min_epochs > 0:
            classifer.fit([X_latent,tics],y,epochs=int(min_epochs),batch_size=batch_size,verbose=0,
                          validation_data=([X_val, tic_val], y_val))

        #train until maximum epochs with early stopping
        classifer.fit([X_latent, tics], y, epochs=int(max_epochs-min_epochs),
                      batch_size=batch_size, verbose=0, callbacks=[cb, history],
                      validation_data=([X_val, tic_val], y_val))

        #get performance
        valLoss = history.history["val_loss"][cb.best_epoch]
        valErr = history.history["val_mean_absolute_error"][cb.best_epoch]
        bestEpoch = cb.best_epoch

        #update best performers
        #if valLoss < bestLoss:
        bestLoss = valLoss
        bestWeights = classifer.get_weights()
        bestValErr = valErr
        bestBestEpoch = bestEpoch
        trainLoss = history.history["loss"][cb.best_epoch]
        trainErr = history.history["mean_absolute_error"][cb.best_epoch]

        #print performance
        print("loss:",trainLoss,"mean_absolute_error:",trainErr,"val loss:", bestLoss, "val_mean_absolute_error:",bestValErr, "numEpochs:", min_epochs + bestBestEpoch)

        #set best weights
        classifer.set_weights(bestWeights)

        self.classifier = classifer

        return history

    def trainClassifierActive(self,X,X_labeled,y_labeled,min_epochs,max_epochs,batch_size,numVal = 10,numManualPerRound=3,inJupyter=True,reinitialize=False):
        trainingInds = []

        valInds = list(range(len(X)))
        valInds = rd.sample(valInds,numVal)

        y = np.zeros((len(X)-numVal,2))

        X_val = X[valInds]
        y_val = np.zeros((numVal,2))

        X = X[[x for x in range(len(X)) if x not in valInds]]

        updatingInds = list(range(len(X)))

        print("label validation peaks:")

        for ind in range(len(X_val)):
            val = self.labelPeak([X_val[ind]], -1*self.windowSize/2, self.windowSize/2, inJupyter,"","relative retention time")
            y_val[ind, 0] = 1 - val
            y_val[ind, 1] = val


        if len(X_labeled) > 0:
            self.trainClassifier(X_labeled,
                                 y_labeled,
                                 X_val,y_val, min_epochs,max_epochs, batch_size, reinitialize=reinitialize)

        y[updatingInds] = self.classifyMatrix(X[updatingInds])

        print("error on validation peaks =", np.mean(np.abs(y_val[:,1] - y[valInds,1])))

        plt.figure()
        plt.hist(y[:, 1], bins=20)
        plt.xlabel("PeakDetective Score")
        plt.ylabel("Number of features")
        plt.show()

        print(str(len(updatingInds)) + " unclassified features remaining")
        print("Continue with active learning? (1=Yes, 0=No): ")
        tmp = float(input())
        while not validateInput(tmp):
            print("invalid classification: ")
            tmp = float(input())

        doMore = bool(tmp)

        i=0

        def padVal(val):
            if val < 0.5:
                return val + 1e-8
            else:
                return val - 1e-8

        while doMore:
            if len(updatingInds) > 0:

                entropies = [-1 * np.sum([padVal(yyy) * np.log(padVal(yyy)) for yyy in yy]) for yy in y[updatingInds]]

                order = list(range(len(updatingInds)))
                order.sort(key=lambda x: entropies[x], reverse=True)
                order = [updatingInds[x] for x in order]

                if len(order) < numManualPerRound:
                    numManualPerRound = len(order)

                inds = np.random.choice(order,numManualPerRound,replace=False,p=np.array(entropies) / np.sum(entropies))

                #inds = order[:numManualPerRound]

                for ind in inds:
                    val = self.labelPeak([X[ind]], -1*self.windowSize/2, self.windowSize/2, inJupyter, y[ind][1],"relative retention time")
                    y[ind, 0] = 1 - val
                    y[ind, 1] = val
                    trainingInds.append(ind)
                    updatingInds.remove(ind)

                if len(X_labeled) > 0:
                    X_train = np.concatenate((X[trainingInds],X_labeled),axis=0)
                    y_train = np.concatenate((y[trainingInds],y_labeled),axis=0)
                else:
                    X_train = X[trainingInds]
                    y_train = y[trainingInds]

                if len(X_train) > 0:
                    self.trainClassifier(X_train,y_train,X_val,y_val,min_epochs,max_epochs,batch_size,reinitialize=reinitialize)

                y[updatingInds] = self.classifyMatrix(X[updatingInds])

                plt.figure()
                plt.hist(y[:, 1], bins=20)
                plt.title("round" + str(i + 1))
                plt.xlabel("PeakDetective Score")
                plt.ylabel("Number of features")
                plt.show()

                print(str(len(updatingInds)) + " unclassified features remaining")
                print("Continue with another iteration? (1=Yes, 0=No): ")
                tmp = float(input())
                while not validateInput(tmp):
                    print("invalid classification: ")
                    tmp = float(input())
                doMore = bool(tmp)

            else:
                doMore = False
            i += 1

        plotScoringStatistics(self.classifyMatrix(X_val)[:,1],y_val[:,1])

        plt.show()


    def classifyMatrix(self,X):
        # normalize matrix
        X_norm = normalizeMatrix(X)
        peak_areas = np.log10(np.array([np.max([2, integratePeak(x)]) for x in X]))

        X_latent = self.encoder.predict(X_norm)
        y = self.classifier.predict([X_latent, peak_areas])
        return y



    def detectPeaksTargeted(self,raw_datas,peaks,ppmCutoff=10,rtCutoff=0.5,cutoff=0.5, intensityCutoff = None,window=0.05,align=True,detectFrac=0.0,isotope="M+0",plot=None,smooth=False):

        #find peaks in targeted list
        rois = peaks[peaks["isotope"] == isotope]["mz"].values

        peak_scores,peak_intensities = self.detectPeaksFromROIs(raw_datas,rois, cutoff, intensityCutoff,window,align,detectFrac,plot=None,smooth=False)

        #look for closest match
        founds = []
        mzs = []
        rts = []
        for index,row in peaks.iterrows():
            if row["isotope"] == isotope:
                filt = peak_scores[(peak_scores["rt"] > row["rt"] - rtCutoff) & (peak_scores["rt"] < row["rt"] + rtCutoff)]
                if len(filt) > 0:
                    minMz = row["mz"] - ppmCutoff/1e6 * row["mz"]
                    maxMz = row["mz"] + ppmCutoff/1e6 * row["mz"]
                    filt = filt[(filt["mz"] > minMz) & (filt["mz"] < maxMz)]
                    if len(filt) > 0:
                        filt["rt_diff"] = [np.abs(row["rt"] - rt) for rt in filt["rt"]]
                        filt["ppm_error"] = [1e6 * np.abs(row["mz"] - mz)/row["mz"] for mz in filt["mz"]]
                        filt = filt.sort_values(by="rt_diff",ascending=True)

                        mzError = filt["mz"].values[0] - row["mz"]

                        peaks_filt = peaks[peaks["group"] == row["group"]]

                        for index2,row2 in peaks_filt.iterrows():
                            founds.append(index2)
                            mzs.append(row2["mz"] + mzError)
                            rts.append(filt["rt"].values[0])


        peaks = peaks.loc[founds,:]
        peaks["mz_searched"] = peaks["mz"]
        peaks["rt_searched"] = peaks["rt"]
        peaks["mz"] = mzs
        peaks["rt"] = rts

        X = self.makeDataMatrix(raw_datas, mzs, rts, align=align,groups=peaks["group"])

        y = self.classifyMatrix(X)

        peak_curated = deepcopy(peaks)
        peak_scores = deepcopy(peaks)
        peak_intensities = deepcopy(peaks)

        keys = []
        for raw in raw_datas:
            peak_scores[raw.filename] = np.zeros(len(peak_scores.index.values))
            peak_intensities[raw.filename] = np.zeros(len(peak_scores.index.values))
            for index in peaks.index.values:
                keys.append([raw.filename, index])

        for [file, index], score in zip(keys, y[:, 1]):
            peak_scores.at[index, file] = score
            val = 0
            if score > cutoff:
                val = 1
            peak_curated.at[index, file] = val

        if "group" in peaks.columns.values:
            peak_groups = peaks[["group"]]
        else:
            peak_groups = None

        print(len(set(peak_groups["group"].values)),"peaks found")

        peak_intensities = self.performIntegration(X, [raw.filename for raw in raw_datas], peak_scores, cutoff,peakGrouping=peak_groups,plot=plot,smooth=smooth)

        return peaks, peak_curated, peak_scores, peak_intensities

    def detectPeaksFromROIs(self, rawDatas, rois, cutoff=0.5, intensityCutoff = None,window=0.05,align=True,detectFrac=0.0,plot=None,smooth=False):
        print("generating all EICs from ROIs...")
        dt = self.windowSize / self.resolution
        tmpRes = int(math.ceil((rawDatas[0].rts[-1] - rawDatas[0].rts[0] + self.windowSize) / dt))
        oldRes = int(self.resolution)
        oldwindow = float(self.windowSize)
        self.resolution = tmpRes
        self.windowSize = rawDatas[0].rts[-1] + self.windowSize / 2 - rawDatas[0].rts[0] + self.windowSize / 2
        rts = [(rawDatas[0].rts[-1] + rawDatas[0].rts[0]) / 2 for _ in rois]
        X_tot = self.makeDataMatrix(rawDatas, rois, rts, align)
        self.resolution = oldRes
        self.windowSize = oldwindow

        numPoints = 0
        rt = rawDatas[0].rts[0]
        while (rt <= rawDatas[0].rts[-1]):
            numPoints += 1
            rt += window

        X = np.zeros((int(numPoints * len(rois) * len(rawDatas)), self.resolution))

        mzs = []
        rts = []
        files = []
        featInds = []

        counter = 0
        rowCounter = 0
        trueDt = tmpRes / (rawDatas[0].rts[-1] + self.windowSize / 2 - rawDatas[0].rts[0] + self.windowSize / 2)
        stride = int(np.floor(window * trueDt))

        noise = []

        for rawData in rawDatas:
            i = 0
            if intensityCutoff is None:
                nL = rawData.calculateNoise()
            else:
                nL = intensityCutoff
            for row in range(len(rois)):
                rt = float(rawDatas[0].rts[0])
                start = 0
                end = start + self.resolution
                for _ in range(numPoints):
                    if end >= X_tot.shape[1]:
                        n = X_tot.shape[1] - start
                        print(start, end, n, X.shape, X_tot.shape, numPoints)
                        X[counter, :n] = X_tot[rowCounter, start:]
                    else:
                        X[counter, :] = X_tot[rowCounter, start:end]
                    counter += 1
                    start += stride
                    end += stride
                    mzs.append(rois[row])
                    rts.append(rt)
                    noise.append(nL)
                    rt += window
                    files.append(rawData.filename)
                    featInds.append(i)
                    i += 1
                rowCounter += 1

        X = X[:counter]

        print(len(X), "EICs constructed for evaluation")

        peak_areas = [integratePeak(x) for x in X]

        print("calculated peak areas")

        toClassify = [x for x in range(len(peak_areas)) if peak_areas[x] > noise[x]]

        print(len(toClassify), "EICs above intensity cutoff")

        peak_scores = pd.DataFrame(index=range(len(set(featInds))))

        orderedMzs = []
        orderedRts = []

        for row in range(len(rois)):
            rt = float(rawDatas[0].rts[0])
            for _ in range(numPoints):
                orderedMzs.append(rois[row])
                orderedRts.append(rt)
                rt += window

        peak_scores["mz"] = orderedMzs
        peak_scores["rt"] = orderedRts

        for rawData in rawDatas:
            peak_scores[rawData.filename] = 0.0


        print("constructed peak scores")


        y = np.zeros(len(X))

        print("starting classification")
        y[toClassify] = self.classifyMatrix(X[toClassify])[:, 1]
        print("done")

        goodInds = []
        for id, filename, score in zip(featInds, files, y):
            if score > cutoff:
                goodInds.append(id)
            peak_scores.at[id, filename] = score

        detectNum = int(np.ceil(len(rawDatas) * detectFrac))
        counts = {x: 0 for x in list(set(goodInds))}
        for x in goodInds:
            counts[x] += 1
        goodInds = [x for x in counts if counts[x] >= detectNum]
        goodInds.sort()

        toKeep = []
        for x in range(len(rawDatas)):
            for g in goodInds:
                toKeep.append(x * len(peak_scores) + g)

        X = X[toKeep]

        peak_scores = peak_scores.loc[goodInds, :]

        peak_scores = peak_scores.reset_index()

        apex_rts = self.updateRT(peak_scores, [raw.filename for raw in rawDatas], X, cutoff)

        peak_scores["rt"] = np.round(apex_rts, 2)

        apex_mzs = self.updateMz(peak_scores, rawDatas, cutoff)

        peak_scores["mz"] = np.round(apex_mzs, 7)

        #peak_intensities = self.performIntegration(X, [raw.filename for raw in rawDatas], peak_scores, cutoff,plot=plot,smooth=smooth)

        #peak_intensities = peak_intensities.drop_duplicates(["mz", "rt"], keep="first")
        peak_scores = peak_scores.drop_duplicates(["mz", "rt"], keep="first")[["mz","rt"]]

        print(len(peak_scores), "peaks putatively detected")

        peak_curated, peak_scores, peak_intensities = self.curatePeaks(rawDatas,peak_scores[["mz","rt"]],cutoff,align,plot,smooth)

        files = list(set(files))
        toDrop = [index for index,row in peak_curated.iterrows() if row[files].sum() < detectNum]

        print(len(toDrop),"peaks failed curation")

        peak_scores = peak_scores.drop(toDrop,axis=0)
        peak_intensities = peak_intensities.drop(toDrop,axis=0)

        print(len(peak_scores), " peaks found")

        return peak_scores, peak_intensities

    def curatePeaks(self,raw_datas,peaks,threshold=0.5,align=False,plot_integration=None,smooth=False):
        print("generating EICs...")
        mzs = peaks["mz"].values
        rts =  peaks["rt"].values

        X = self.makeDataMatrix(raw_datas,mzs,rts,align=align)

        y = self.classifyMatrix(X)

        peak_curated = deepcopy(peaks)
        peak_scores = deepcopy(peaks)
        peak_intensities = deepcopy(peaks)

        keys = []
        for raw in raw_datas:
            peak_scores[raw.filename] = np.zeros(len(peak_scores.index.values))
            peak_intensities[raw.filename] = np.zeros(len(peak_scores.index.values))
            for index in peaks.index.values:
                keys.append([raw.filename, index])

        for [file, index], score in zip(keys, y[:, 1]):
            peak_scores.at[index, file] = score
            val = 0
            if score > threshold:
                val = 1
            peak_curated.at[index, file] = val

        peak_intensities = self.performIntegration(X, [raw.filename for raw in raw_datas], peak_scores, threshold,plot=plot_integration,smooth=smooth)

        return peak_curated,peak_scores,peak_intensities

    def detectPeaks(self, rawDatas, cutoff=0.5, intensityCutoff = None,numDataPoints=3,window=0.05,align=True,detectFrac=0.0,plot=None,smooth=False):
        rois = self.roiDetection(rawDatas, intensityCutoff=intensityCutoff, numDataPoints=numDataPoints)
        peak_scores,peak_intensities = self.detectPeaksFromROIs(rawDatas,rois, cutoff, intensityCutoff ,window,align,detectFrac,plot,smooth=smooth)
        return peak_scores,peak_intensities,rois

    def updateRT(self,peakScores,samples,X,cutoff,smooth=False):
        i = 0
        rts = []
        dt = self.windowSize / self.resolution
        for index, row in peakScores.iterrows():
            inds = []
            allInds = []
            for n,samp in enumerate(samples):
                ind = i + n * len(peakScores)
                if row[samp] > cutoff:
                    inds.append(ind)
                allInds.append(ind)
            if len(inds) > 0:
                tmp = X[inds].sum(axis=0)
                if smooth: tmp = self.smoother.predict(normalizeMatrix(np.array([tmp])), verbose=0)[0]
                lb, rb, apex = findPeakBoundaries(tmp)

            else:
                apex = (self.resolution-1)/2

            rt = (apex - (self.resolution-1)/2) * dt + row["rt"]
            rts.append(rt)
            i += 1
            printProgressBar(i, len(peakScores), "refining retention times", printEnd="")

        return rts

    def updateMz(self,peak_scores,raw_data,cutoff):
        mzs = []
        i = 0
        for index,row in peak_scores.iterrows():
            tmp = []
            for data in raw_data:
                if row[data.filename] > cutoff:
                    tmp.append(data.getApexMz(row["mz"],row["rt"]))
            tmp = np.array(tmp)
            mzs.append(np.average(tmp[:,0],weights=tmp[:,1]))
            i += 1
            printProgressBar(i, len(peak_scores), "refining mzs", printEnd="")
        return mzs

    def label_peaks(self,raw_data,peaks,inJupyter = True):
        rt_starts = [row["rt"] - self.windowSize/2 for _,row in peaks.iterrows()]
        rt_ends = [row["rt"] + self.windowSize/2 for _,row in peaks.iterrows()]
        y = []
        mat = self.makeDataMatrix([raw_data],peaks["mz"].values,peaks['rt'])
        count = 1
        for vec,rt_start,rt_end in zip(mat,rt_starts,rt_ends):
            y.append(self.labelPeak([vec],rt_start,rt_end,inJupyter,str(count) + "/" + str(len(mat)),"retention time"))
            count += 1
        peaks["classification"] = y
        return peaks

    def labelPeak(self,vecs,rt_start,rt_end,inJupyter,title="",xlabel="retention time"):
        plt.figure()
        plt.ion()
        xs = np.linspace(rt_start, rt_end, len(vecs[0]))
        [plt.plot(xs, vec) for vec in vecs]
        plt.xlabel(xlabel)
        plt.ylabel("intensity")
        plt.title(title)
        plt.show(block=False)
        plt.pause(0.001)
        print("Enter classification (1=True Peak, 0=Artifact): ")
        val = input()

        while not validateInput(val):
            print("invalid classification: ")
            val = input()
        val = float(val)
        plt.close()
        if inJupyter:
            IPython.display.clear_output(wait=True)

        return val

    def roiDetection(self,rawdatas,intensityCutoff=None,numDataPoints = 3):
        rtss = [rawdata.rts for rawdata in rawdatas]

        roises = startConcurrentTask(ROISearch,[[raw_data,intensityCutoff,numDataPoints] for raw_data in rawdatas],np.min([self.numCores,len(rawdatas)]),"running ROI search",len(rawdatas))

        roises = [{x:100 for x in rois} for rois in roises]

        ppm = np.median([f.ppm for f in rawdatas])

        rois = list(mergeSpectra(roises,ppm).keys())

        rois.sort()

        print()
        print(len(rois)," ROIs found")

        return rois

    def filterPeakListByScores(self,peakScores,samples,cutoff,frac):
        goodInds = [index for index,row in peakScores.iterrows() if float(len([x for x in samples if row[x] > cutoff])) / len(samples) > frac]
        return peakScores.loc[goodInds,:]

    def performIntegration(self, X, samples, peakScores, cutoff, defaultWidth=0.5,smooth=False,peakGrouping=None,plot=None):

        if not plot is None and type(plot) == type(""):
            try:
                os.mkdir(plot)
            except:
                pass

        if peakGrouping is None:
            peakGrouping = pd.DataFrame(index=peakScores.index.values)
            peakGrouping["group"] = [x for x in range(len(peakGrouping))]

        mapper = {index:i for index,i in zip(peakScores.index.values,range(len(peakScores)))}

        peak_areas = pd.DataFrame(index=peakScores.index.values,columns=["mz","rt"])
        peak_areas["mz"] = peakScores["mz"].values
        peak_areas["rt"] = peakScores["rt"].values
        peak_areas[samples] = np.zeros(peakScores[samples].values.shape)
        print("integrating peaks...")

        integ_groups = list(set(peakGrouping["group"].values))
        counter = 0
        for g in integ_groups:
            filt = peakScores.loc[peakGrouping[peakGrouping["group"] == g].index.values,:]
            rt = filt["rt"].values[0]
            inds = []
            allInds = []
            indexList = []
            sampList = []
            for index,row in filt.iterrows():
                i = mapper[index]
                for n, samp in enumerate(samples):
                    ind = i + n * len(peakScores)
                    if row[samp] > cutoff:
                        inds.append(ind)
                    allInds.append(ind)
                    sampList.append(samp)
                    indexList.append(index)
            if len(inds) > 0:
                tmp = X[inds].sum(axis=0)
                if smooth: tmp = self.smoother.predict(normalizeMatrix(np.array([tmp])),verbose=0)[0]
                lb,rb,apex = findPeakBoundaries(tmp)
                if lb == rb:
                    lb = int(np.round(self.resolution / 2 - defaultWidth * self.resolution / 2))
                    rb = int(np.round(self.resolution / 2 + defaultWidth * self.resolution / 2))
            else:
                lb = int(np.round(self.resolution / 2 - defaultWidth * self.resolution / 2))
                rb = int(np.round(self.resolution/2 + defaultWidth * self.resolution/2))
                tmp = np.zeros(X.shape[1])

            for index,ind,sample in zip(indexList,allInds,sampList):
                peak_areas.at[index,sample] = integratePeak(X[ind],[lb,rb])

            if not plot is None and type(plot) == type(""):

                xaxis = np.linspace(rt-self.windowSize/2,rt + self.windowSize/2,X.shape[1])

                norm = np.max(X[allInds])

                #normalize tmp
                tmp = norm * safeNormalizeMax(tmp)

                #maxV = np.max(X[allInds].flatten())
                #tmp = (tmp / np.max(tmp)) * maxV

                plt.figure()
                for ind in inds:#allInds:
                    plt.plot(xaxis,X[ind],alpha=0.5,color="green")
                for ind in allInds:
                    if ind not in inds:
                        plt.plot(xaxis,X[ind],alpha=0.5,color="grey")
                if len(inds) > 0:
                    plt.plot(xaxis, tmp, color="red", linestyle="--")

                #plt.plot(xaxis,tmp,color="red",linestyle="--")
                plt.plot([xaxis[lb],xaxis[lb]],[0,norm],color="grey",linestyle="--")
                plt.plot([xaxis[rb],xaxis[rb]],[0,norm],color="grey",linestyle="--")
                plt.xlabel("RT")
                plt.ylabel("Intensity")
                plt.savefig(plot + "/" + str(g) + ".png")
                plt.close()

            counter += 1
            printProgressBar(counter, len(integ_groups), "integrating peaks", printEnd="")

        return peak_areas


class rawData():
    def __init__(self,data={},filename="",ppm=0,timestamp=None):
        self.data = data
        self.filename = filename
        self.rts = list(self.data.keys())
        self.rts.sort()
        self.ppm = ppm
        self.timestamp = None

    def readRawDataFile(self,filename,ppm,intensityThresh = 0):
        """
         Read MS datafile

        :param filename: str, path to MS datafile
        """
        try:
            try:
                with mzml.MzML(filename.replace('"', "")) as f:
                    self.timestamp = datetime.datetime.fromisoformat(next(f.iterfind('run', recursive=False))['startTimeStamp'][:-1]).astimezone(datetime.timezone.utc)
            except:
                print("Warning: file timestamp could not be read")
                self.timestamp = None
            reader = mzml.read(filename.replace('"', ""))
            ms1Scans = {}
            for temp in reader:
                if temp['ms level'] == 1:
                    spectrum = [[mz,i] for mz, i in zip(temp["m/z array"],temp["intensity array"]) if i > intensityThresh]
                    spectrum.sort(key=lambda x:x[0])
                    ms1Scans[temp["scanList"]["scan"][0]["scan start time"]] = spectrum
            reader.close()
            self.rts = list(ms1Scans.keys())
            self.rts.sort()
            self.data = ms1Scans
            self.filename = filename
            self.ppm = ppm

        except:
            print(sys.exc_info())
            print(filename + " does not exist or is ill-formatted")


    def extractEIC(self,mz,rt_start,rt_end):
        width = self.ppm * mz / 1e6
        mz_start = mz - width
        mz_end = mz + width
        rts = [x for x in self.rts if x > rt_start and x < rt_end]
        intensity = []
        for rt in rts:
            Origind = getIndexOfClosestValue([x[0] for x in self.data[rt]],mz)
            ind = int(Origind)
            tmp = 0
            while  ind < len(self.data[rt]) and self.data[rt][ind][0] > mz_start and self.data[rt][ind][0] < mz_end:
                tmp += self.data[rt][ind][1]
                ind += 1
            ind = Origind - 1
            while ind > -1 and self.data[rt][ind][0] > mz_start and self.data[rt][ind][0] < mz_end :
                tmp += self.data[rt][ind][1]
                ind -= 1
            intensity.append(tmp)
        return rts,intensity

    def getApexMz(self,mz,rt):
        width = self.ppm * mz / 1e6
        mz_start = mz - width
        mz_end = mz + width
        rt = self.rts[getIndexOfClosestValue(self.rts,rt)]
        highestMz = mz
        highestIntensity = 1
        Origind = getIndexOfClosestValue([x[0] for x in self.data[rt]], mz)
        ind = int(Origind)
        while ind < len(self.data[rt]) and self.data[rt][ind][0] > mz_start and self.data[rt][ind][0] < mz_end:
            if self.data[rt][ind][1] > highestIntensity:
                highestIntensity = self.data[rt][ind][1]
                highestMz = self.data[rt][ind][0]
            ind += 1
        ind = Origind - 1
        while ind > -1 and self.data[rt][ind][0] > mz_start and self.data[rt][ind][0] < mz_end:
            if self.data[rt][ind][1] > highestIntensity:
                highestIntensity = self.data[rt][ind][1]
                highestMz = self.data[rt][ind][0]
            ind -= 1
        return highestMz,highestIntensity

    def interpolate_data(self,mz,rt_start,rt_end):
        rts,intensity = self.extractEIC(mz,rt_start,rt_end)
        if len(rts) > 3:
            #smoothing = smoothing * len(rts) * np.max(intensity)
            #s = UnivariateSpline(rts,intensity,ext=1,s=smoothing)
            s = interp1d(rts,intensity,kind="linear",fill_value=0,bounds_error=False)
        else:
            #s = UnivariateSpline([0,5,10,15],[0,0,0,0],ext=1,s=smoothing)
            s = interp1d([0,5,10,15],[0,0,0,0],kind="linear",fill_value=0,bounds_error=False)

        return s

    def getMergedSpectrum(self,rtRange=None,intensityThresh=0,ppm=1,n=None):
        if rtRange is None:
            rtRange = [self.rts[0],self.rts[-1]]
        spectra = [{mz:i for mz,i in self.data[rt] if i > intensityThresh} for rt in self.rts if rt > rtRange[0] and rt < rtRange[1]]
        if not n is None:
            if n < len(spectra):
                print(len(spectra),'spectra')
                spectra = rd.sample(spectra,k=n)
        return mergeSpectra(spectra,ppm)

    def calculateNoise(self):
        merged = self.getMergedSpectrum(n=500)

        vals = np.array(list(merged.values()))

        mix = GaussianMixture(2)

        mix.fit(np.log10(vals).reshape(-1, 1))

        xs = np.linspace(np.min(np.log10(vals)),np.max(np.log10(vals)),1000)

        if mix.means_[0] < mix.means_[1]:
            false_positive = area_under_gaussian_at_right(xs, mix.means_[0], np.sqrt(mix.covariances_[0][0]))
            false_negative = area_under_gaussian_at_left(xs, mix.means_[1], np.sqrt(mix.covariances_[1][0]))

        else:
            false_positive = area_under_gaussian_at_right(xs, mix.means_[1], np.sqrt(mix.covariances_[1][0]))
            false_negative = area_under_gaussian_at_left(xs, mix.means_[0], np.sqrt(mix.covariances_[0][0]))

        threshold = xs[np.argmin(false_negative + false_positive)]

        return 2*np.power(10,threshold)



def Smoother(resolution):
    # build autoencoder
    autoencoderInput = keras.Input(shape=(resolution,))
    x = layers.Reshape((resolution, 1))(autoencoderInput)

    kernelsize = 3
    stride = 1
    max_norm_value = 2.0

    x = layers.Conv1D(32, kernelsize, strides=stride, activation='relu', kernel_constraint=max_norm(max_norm_value),
                     kernel_initializer='he_uniform')(x)

    #x = layers.BatchNormalization()(x)

    x = layers.Conv1D(16, kernelsize, strides=stride, activation='relu', kernel_constraint=max_norm(max_norm_value),
                     kernel_initializer='he_uniform')(x)

    #x = layers.BatchNormalization()(x)

    x = layers.Conv1D(8, kernelsize, strides=stride, activation='relu', kernel_constraint=max_norm(max_norm_value),
                      kernel_initializer='he_uniform')(x)

    #x = layers.BatchNormalization()(x)

    x = layers.Conv1D(4, kernelsize, strides=stride, activation='relu', kernel_constraint=max_norm(max_norm_value),
                      kernel_initializer='he_uniform')(x)

    #x = layers.BatchNormalization()(x)

    x = layers.Flatten()(x)

    x = layers.Dense(5, activation="relu")(x)

    x = layers.Dense(int((resolution-8) * 4), activation="relu")(x)

    x = layers.Reshape((resolution-8, 4))(x)

    x = layers.Conv1DTranspose(8, kernelsize, strides=stride, activation='relu',
                               kernel_constraint=max_norm(max_norm_value), kernel_initializer='he_uniform')(x)

    #x = layers.BatchNormalization()(x)

    x = layers.Conv1DTranspose(16, kernelsize, strides=stride, activation='relu',
                              kernel_constraint=max_norm(max_norm_value), kernel_initializer='he_uniform')(x)

    #x = layers.BatchNormalization()(x)

    x = layers.Conv1DTranspose(32, kernelsize, strides=stride, activation='relu',
                              kernel_constraint=max_norm(max_norm_value), kernel_initializer='he_uniform')(x)

    #x = layers.BatchNormalization()(x)

    x = layers.Conv1DTranspose(1, kernelsize, strides=stride, activation='sigmoid',
                               kernel_constraint=max_norm(max_norm_value), kernel_initializer='he_uniform')(x)

    x = layers.Flatten()(x)

    autoencoder = keras.Model(autoencoderInput, x)

    autoencoder.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(1e-4),
                        metrics=['mean_absolute_error'],weighted_metrics=[])


    return autoencoder

def ClassifierLatent(resolution):
    descriminatorInput = keras.Input(shape=(resolution,))
    ticInput = keras.Input(shape=(1,))

    #x = layers.Dense(resolution, activation="relu")(descriminatorInput)

    x = layers.Layer()(descriminatorInput)

    x = keras.Model(descriminatorInput, x)

    tic = keras.Model(ticInput, layers.Layer()(ticInput))

    x = layers.concatenate([x.output, tic.output], axis=1)

    x = layers.Dense(int(resolution), activation="relu")(x)

    x = layers.Dense(int(resolution), activation="relu")(x)

    x = layers.Dense(int(resolution), activation="relu")(x)

    x = layers.Dense(int(resolution), activation="relu")(x)

    x = layers.Dense(int(resolution), activation="relu")(x)

    output = layers.Dense(2, activation="softmax")(x)

    classifier = keras.Model([descriminatorInput, ticInput], output, name="discriminator")

    classifier.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(1e-4),
                          metrics=['mean_absolute_error'])

    return classifier


def makePRCPlot(pred,true,noSkill=True):

    prec, recall, threshs = met.precision_recall_curve(true, pred)

    auc = np.round(met.auc(recall, prec), 4)

    plt.plot(recall, prec, label="prAUC=" + str(auc))
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    if noSkill:
        numPositive = len([x for x in true if x > 0.5])
        numNegative = len(true) - numPositive
        plt.plot([0, 1.0],
                 [numPositive / float(numPositive + numNegative), numPositive / float(numPositive + numNegative)],
                 label="NSL prAUC=" + str(
                     np.round(numPositive / float(numPositive + numNegative), 4)))
    plt.legend()
    return auc


def findPeakBoundaries(peak):

    peaks,_ = find_peaks(peak)

    if len(peaks) > 0:

        peaks = list(peaks)

        peaks.sort(key=lambda x: abs(x-int(len(peak)/2)))

        peaks = peaks[:1]

        half_widths = peak_widths(peak,peaks,rel_height=0.5)[0][0]

        lb = int(np.round(peaks[0] - (half_widths/2)))

        rb = int(np.round(peaks[0] + (half_widths/2)))

        apex = peaks[0]


    else:
        lb = int(len(peak)/2)-3
        rb = int(len(peak)/2)+3
        apex = int(len(peak)/2)

    return np.max([0,lb]),np.min([len(peak)-1,rb]),apex

def integratePeak(peak,bounds=None):
    if bounds is None:
        lb,rb,apex = findPeakBoundaries(peak)
    else:
        lb = bounds[0]
        rb = bounds[1]
    if lb != rb:
        try:
            area = np.trapz(peak[lb:rb],np.linspace(lb,rb,rb-lb))
        except:
            print(lb,rb)
            area = 0
    else:
        area = 0
    return area


def alignPeaks(peaks,normalize=True,reference=0,sampInds = None,q=None):

    if sampInds is None:
        sampInds = list(range(len(peaks)))

    uniqueSamps = list(set(sampInds))

    uniqueSamps.sort()

    sampMapper = {samp:[x for x in range(len(peaks)) if sampInds[x] == samp] for samp in uniqueSamps}

    composite_peaks = [np.sum(np.array(peaks)[sampMapper[x]],axis=0) for x in uniqueSamps]

    if normalize:
        peaks_norm = [safeNormalize(x) for x in composite_peaks]
    else:
        peaks_norm = deepcopy(composite_peaks)

    reference_peak = peaks_norm[reference]
    ref = list(range(len(reference_peak)))

    for x,peak in enumerate(peaks_norm):
        if x > 0:
            distance, path = fastdtw(np.array(reference_peak).flatten(), np.array(peak).flatten(), dist=2)
            xs = []
            ys = []

            prev = path[0][0]
            tmp = [[peaks[y][path[0][1]] for y in sampMapper[x]]]
            for p1,p2 in path[1:]:
                if p1 != prev:
                    xs.append(prev)
                    ys.append(np.mean(tmp,axis=1))
                    tmp = []
                tmp.append([peaks[y][p2] for y in sampMapper[x]])
                prev = p1
            xs.append(prev)
            ys.append(np.mean(tmp,axis=1))

            ys = np.array(ys)


            for i in range(ys.shape[1]):
                f = interp1d(xs,ys[:,i],fill_value=0.0,bounds_error=False)
                peaks[sampMapper[x][i]] = [f(i) for i in ref]


    if type(q) != type(None):
        q.put(0)

    return peaks

def alignDataMatrix(matrix,peakInds,sampInds,normalize=True,numCores=1):

    #gather list of matrix rows that correspond to peak groups for all samples
    #peak inds should indicate a unique peak
    #sampe inds should indicate which peaks below to which samples

    uniquePeaks = list(set(peakInds))
    peakMap = {peak:i for i,peak in enumerate(uniquePeaks)}
    order = [[] for _ in uniquePeaks]
    args = [[[],normalize,0,[]] for _ in uniquePeaks]
    for i,peak,ind,sampInd in zip(range(len(matrix)),matrix,peakInds,sampInds):
        args[peakMap[ind]][0].append(peak)
        args[peakMap[ind]][3].append(sampInd)
        order[peakMap[ind]].append(i)
    result = startConcurrentTask(alignPeaks,args,numCores,"aligning EICs",len(uniquePeaks))
    for peaks,inds in zip(result,order):
        for peak,ind in zip(peaks,inds):
            matrix[ind] = peak
    return matrix


def takeClosestInd(myList, myNumber):
    """
    Assumes myList is sorted. Returns closest value to myNumber.

    If two numbers are equally close, return the smallest number.
    """

    pos = bisect_left(myList, myNumber)
    if pos == 0:
        return pos
    if pos == len(myList):
        return len(myList) - 1
    before = myList[pos - 1]
    after = myList[pos]
    if after - myNumber < myNumber - before:
        return pos
    else:
        return pos-1

def mergeSpectra(spectra,ppm):
    if len(spectra) > 0:
        mergedSpectrumMzs = [x for x in spectra[0]]
        mergedSpectrumMzs.sort()
        mergedSpectrumInts = [spectra[0][x] for x in mergedSpectrumMzs]

        if len(spectra) > 1:
            fact = ppm / 1e6
            for spectrum in spectra[1:]:
                for mz,i in spectrum.items():

                    if len(mergedSpectrumMzs) > 0:

                        x = takeClosestInd(mergedSpectrumMzs,mz)
                        delta = mz * fact

                        if mergedSpectrumMzs[x] > mz - delta and mergedSpectrumMzs[x] < mz + delta:
                            mergedSpectrumInts[x] += i
                        else:
                            if mz < mergedSpectrumMzs[x]:
                                mergedSpectrumMzs.insert(x,mz)
                                mergedSpectrumInts.insert(x,i)
                            else:
                                mergedSpectrumMzs.insert(x+1,mz)
                                mergedSpectrumInts.insert(x+1,i)

                    else:
                        mergedSpectrumMzs.append(mz)
                        mergedSpectrumInts.append(i)
    else:
        mergedSpectrumMzs = []
        mergedSpectrumInts = []

    return {mz:i for mz,i in zip(mergedSpectrumMzs,mergedSpectrumInts)}

def ROISearch(rawdata,intensityCutoff,numDataPoints,q=None):
    if intensityCutoff is None:
        iC = rawdata.calculateNoise()
    else:
        iC = intensityCutoff

    ppm = rawdata.ppm
    rois = []
    for rt in rawdata.data:
        for mz, i in rawdata.data[rt]:
            if i > iC:
                update, pos = binarySearchROI(rois, mz, ppm)
                if update:
                    if not rois[pos]["closed"]:
                        rois[pos]["mzs"].append(mz)
                        rois[pos]["mz_mean"] = np.mean(rois[pos]["mzs"])
                        rois[pos]["extended"] = True
                        rois[pos]["count"] += 1
                        if rois[pos]["count"] >= numDataPoints:
                            rois[pos]["closed"] = True
                else:
                    if pos != len(rois):
                        rois.insert(pos, {"mz_mean": mz, "mzs": [mz], "extended": True, "count": 1, "closed": False})
                    else:
                        rois.append({"mz_mean": mz, "mzs": [mz], "extended": True, "count": 1, "closed": False})

        toKeep = []
        for x in range(len(rois)):
            if rois[x]["extended"] == True or rois[x]["closed"]:
                toKeep.append(x)

        rois = [rois[x] for x in toKeep]

        for x in range(len(rois)):
            rois[x]["extended"] = False

    if not q is None:
        q.put([])

    return [x["mz_mean"] for x in rois if x["closed"]]


#Utility functions:

#runs a function concurently.
def startConcurrentTask(task,args,numCores,message,total,chunksize="none",verbose=True):

    if verbose:
        m = Manager()
        q = m.Queue()
        args = [a + [q] for a in args]
        t = Thread(target=updateProgress, args=(q, total, message))
        t.start()
    if numCores > 1:
        p = Pool(numCores)
        if chunksize == "none":
            res = p.starmap(task, args)
        else:
            res = p.starmap(task, args, chunksize=chunksize)
        p.close()
        p.join()
    else:
        res = [task(*a) for a in args]
    if verbose: t.join()
    return res


def safeNormalize(x):
    """
    Safely normalize a vector, x, to sum to 1.0. If x is the zero vector return the normalized unity vector
    """
    if np.sum(x) < 1e-6:
        tmp = np.ones(x.shape)
        return tmp / np.sum(tmp)
    else:
        return x/np.sum(x)

def safeNormalizeMax(x):
    """
    Safely normalize a vector, x, to sum to 1.0. If x is the zero vector return the normalized unity vector
    """
    if np.max(x) < 1e-6:
        tmp = np.zeros(x.shape)
        return tmp
    else:
        return x / np.max(x)

def normalizeMatrix(X):
    """
    Normalize a matrix so that the rows of X sum to one
    """
    return np.array([safeNormalize(x) for x in X])

def classify(preds):
    """
    Classify predictions, preds, by returning the index of the highest scoring class
    """
    classes = np.zeros(preds.shape)
    for x in range(len(preds)):
        classes[x,list(preds[x]).index(np.max(list(preds[x])))] = 1
    return classes


def take_closest(myList, myNumber):
    """
    Assumes myList is sorted. Returns closest value to myNumber.

    If two numbers are equally close, return the smallest number.
    """
    pos = bisect_left(myList, myNumber)
    if pos == 0:
        return myList[0]
    if pos == len(myList):
        return myList[-1]
    before = myList[pos - 1]
    after = myList[pos]
    if after - myNumber < myNumber - before:
        return after
    else:
        return before



def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=50, fill='█', printEnd="\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()

def updateProgress(q, total,message = ""):
    """
    update progress bar
    """
    counter = 0
    while counter != total:
        if not q.empty():
            q.get()
            counter += 1
            #if counter % 20 == 0:
            printProgressBar(counter, total,prefix = message, printEnd="")


def getIndexOfClosestValue(l,v):
    """
    get index of list with closest value to v, assumes l is sorted
    """
    pos = bisect_left(l, v)
    if pos == 0:
        return 0
    if pos == len(l):
        return pos - 1
    before = l[pos - 1]
    after = l[pos]
    if after - v < v - before:
        return pos
    else:
        return pos - 1

def plotScoringStatistics(scores, labels):
    tpr = []
    fdr = []
    cutoffs = np.linspace(0, 1.0, 100)
    for cutoff in cutoffs:
        fdr.append(1 - met.precision_score(labels, scores > cutoff, zero_division=0))
        tpr.append(met.recall_score(labels, scores > cutoff, zero_division=0))
    plt.scatter(cutoffs, tpr, c="black", label="TPR")
    plt.plot(cutoffs, tpr, c="black")

    plt.scatter(cutoffs, fdr, c="red", label="FDR")
    plt.plot(cutoffs, fdr, c="red")
    plt.legend()
    plt.xlabel("cutoff")
    plt.ylabel("performance")

def validateInput(input):
    """
    Validate input, used for the labelPeaks() function
    @param input: str, user input
    @return: Bool, True = user input was a 0 or 1, False otherwise
    """
    try:
        input = float(input)
        if input not in [0, 1]:
            return False
        return True
    except:
        return False

def binarySearchROI(poss, query, ppm):
    """
    Search function used in ROI calculations
    """
    if len(poss) == 0:
        return False, 0

    pos = bisect_left([x["mz_mean"] for x in poss], query)

    if pos == len(poss):
        if 1e6 * np.abs(query - poss[-1]["mz_mean"]) / query < ppm:
            return True, pos - 1
        else:
            return False, pos
    elif pos == 0:
        if 1e6 * np.abs(query - poss[0]["mz_mean"]) / query < ppm:
            return True, 0
        else:
            return False, pos
    else:
        ldiff = 1e6 * np.abs(query - poss[pos - 1]["mz_mean"]) / query
        rdiff = 1e6 * np.abs(query - poss[pos]["mz_mean"]) / query

        if ldiff < rdiff:
            if ldiff < ppm:
                return True, pos - 1
            else:
                return False, pos
        else:
            if rdiff < ppm:
                return True, pos
            else:
                return False, pos

def area_under_gaussian_at_left(t, mu, sigma):
    a = t - mu
    b = math.sqrt(2) * sigma
    return .5 * (1 + scipy.special.erf(a / b))

def area_under_gaussian_at_right(t, mu, sigma):
    return 1 - area_under_gaussian_at_left(t, mu, sigma)