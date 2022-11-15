from pyteomics import mzml
import sys
import numpy as np
from scipy.interpolate import UnivariateSpline
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
import sklearn.metrics as met
import pandas as pd


#Utility functions:
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
    get index of list with closest value to v
    """
    order = list(range(len(l)))
    order.sort(key=lambda x:np.abs(l[x]-v))
    return order[0]

def generateSkylineFiles(peak_scores,peak_boundaries,samples,polarity,moleculeListName = "XCMS peaks"):
    transitionList = deepcopy(peak_scores)

    transitionList["Precursor Name"] = ["unknown " + str(index) for index, row in transitionList.iterrows()]
    transitionList["Explicit Retention Time"] = [row["rt"] for index, row in
                                                 transitionList.iterrows()]
    polMapper = {"Positive": 1, "Negative": -1}
    transitionList["Precursor Charge"] = [polMapper[polarity] for index, row in transitionList.iterrows()]
    transitionList["Precursor m/z"] = [row["mz"] for index,row in transitionList.iterrows()]
    transitionList["Molecule List Name"] = [moleculeListName for _ in range(len(transitionList))]
    transitionList = transitionList[
        ["Molecule List Name", "Precursor Name", "Precursor m/z", "Precursor Charge",
         "Explicit Retention Time"]]

    peakBoundariesSkyline = {}

    # iterate through filenames and cpds
    for fn in samples:
        for index, row in transitionList.iterrows():
            peakBoundariesSkyline[len(peakBoundariesSkyline)] = {"Min Start Time": peak_boundaries.at[index,fn][0],
                                                   "Max End Time": peak_boundaries.at[index,fn][1],
                                                   "Peptide Modified Sequence": row["Precursor Name"],
                                                   "File Name": fn}

    # format as df
    peakBoundariesSkyline = pd.DataFrame.from_dict(peakBoundariesSkyline, orient="index")
    peakBoundariesSkyline = peakBoundariesSkyline[["File Name", "Peptide Modified Sequence", "Min Start Time", "Max End Time"]]

    return transitionList,peakBoundariesSkyline





class PeakDetective():
    """
    Class for curation/detection of LC/MS peaks in untargerted metabolomics data
    """
    def __init__(self,resolution=100,numCores=1,windowSize = 1.0):
        self.resolution = resolution
        self.numCores = numCores
        self.windowSize = windowSize
        self.smoother = Smoother(resolution)
        self.classifier = Classifier(resolution)
        self.encoder = keras.Model(self.smoother.input, self.smoother.layers[7].output)

    def plot_overlayedEIC(self,rawdatas,mz,rt_start,rt_end,smoothing=0,alpha=0.3):
        ts = np.linspace(rt_start,rt_end,self.resolution)
        for data in rawdatas:
            s = data.interpolate_data(mz,rt_start,rt_end,smoothing)
            ints  = [np.max([x,0]) for x in s(ts)]
            plt.plot(ts,ints,alpha=alpha)


    @staticmethod
    def getNormalizedIntensityVector(data,mzs,rtstarts,rtends,smoothing,resolution,q=None):
        out = np.zeros((len(mzs),resolution))
        i=0
        for mz,rt_start,rt_end in zip(mzs,rtstarts,rtends):
            s = data.interpolate_data(mz,rt_start,rt_end,smoothing)
            out[i,:] = s(np.linspace(rt_start,rt_end,resolution))
            i += 1
        if type(q) != type(None):
            q.put(0)
        return out

    def makeDataMatrix(self,rawdatas,mzs,rts,smoothing=0):
        rtstarts = [rt - self.windowSize/2 for rt in rts]
        rtends = [rt + self.windowSize/2 for rt in rts]
        args = []
        numToGetPerProcess = int(len(rawdatas)*len(mzs)/float(self.numCores))
        for rawdata in rawdatas:
            tmpMzs = []
            tmpRtStarts = []
            tmpRTends = []
            for mz,rtstart,rtend in zip(mzs,rtstarts,rtends):
                tmpMzs.append(mz)
                tmpRtStarts.append(rtstart)
                tmpRTends.append(rtend)
                if len(tmpMzs) == numToGetPerProcess:
                    args.append([rawdata,tmpMzs,tmpRtStarts,tmpRTends,smoothing,self.resolution])
                    tmpMzs = []
                    tmpRtStarts = []
                    tmpRTends = []

            if len(tmpMzs) > 0: args.append([rawdata, tmpMzs, tmpRtStarts, tmpRTends, smoothing,self.resolution])

        result = startConcurrentTask(PeakDetective.getNormalizedIntensityVector, args, self.numCores, "forming matrix", len(args))
        return np.concatenate(result,axis=0)

    def generateGaussianPeaks(self,n, centerDist, numPeaksDist=(0,2), widthFactor=0.1, heightFactor=1):
        X_signal = np.zeros((n,self.resolution)) #self.generateNoisePeaks(X_norm, tics)
        switcher = list(range(len(centerDist)))
        for x, s in zip(range(len(X_signal)), np.random.random(len(X_signal))):
            if numPeaksDist[0] >= numPeaksDist[1]:
                numGuass = numPeaksDist[0]
            else:
                numGuass = np.random.randint(numPeaksDist[0], numPeaksDist[1])
            for _ in range(numGuass):
                ind = int(np.random.choice(switcher))
                X_signal[x] += heightFactor * stats.norm.pdf(np.linspace(0, 1, self.resolution),
                                     stats.uniform.rvs(centerDist[ind][0], centerDist[ind][1] - centerDist[ind][0]),
                                     widthFactor * s + .001)
            #if np.sum(tmp) > 0:
            #    tmp = (1 - s2n[x]) * tmp / np.sum(tmp)
        X_signal = normalizeMatrix(X_signal)

        return X_signal

    def generateFalsePeaks(self,peaks,raw_datas, n=None):
        if type(n) == type(None):
            n = len(peaks)

        peaks_rand = pd.DataFrame()
        peaks_rand["rt"] = rd.choices(list(peaks["rt"].values),k=n)
        peaks_rand["mz"] = rd.choices(list(peaks["mz"].values),k=n)

        X_noise = self.makeDataMatrix(raw_datas,peaks_rand["mz"].values,peaks_rand["rt"].values)

        return X_noise

    def generateSignalPeaks(self,peaks,raw_datas,widthFactor = 0.1,heightFactor = 1,n=None):
        if type(n) == type(None):
            n = len(peaks)
        X_noise = self.generateFalsePeaks(peaks,raw_datas,n=n)
        X_signal = self.generateGaussianPeaks(int(n*len(raw_datas)),[[0.45,0.5],[0.5,0.55]],(1,1),widthFactor,heightFactor)

        samp = peaks.loc[rd.choices(list(peaks.index.values),k=int(np.ceil(n))),:]
        mzs = list(samp["mz"].values)

        tmp = self.makeDataMatrix(raw_datas,mzs,samp["rt"].values)
        tmp = tmp[:n*len(raw_datas),:]

        signal_areas = np.array([integratePeak(x) for x in tmp])

        X_signal = signal_areas[:, np.newaxis] * X_signal

        X = X_noise + X_signal

        return X


    def trainSmoother(self,peaks,raw_datas,numPeaks,smooth_epochs,batch_size,validation_split):
        #generate data matrix
        print("generating EICs...")
        mzs = rd.choices(list(peaks["mz"].values),k=int(numPeaks/len(raw_datas)))
        rts = rd.choices(list(peaks["rt"].values),k=int(numPeaks/len(raw_datas)))
        rt_starts = [rt - self.windowSize/2 for rt in rts]
        rt_ends = [rt + self.windowSize/2 for rt in rts]

        X = self.makeDataMatrix(raw_datas,mzs,rts)

        #normalize matrix
        X_norm = normalizeMatrix(X)

        print("done")

        #fit autoencoder
        print("fitting smoother...")
        smoother = Smoother(self.resolution)
        smoother.fit(X_norm, X_norm, epochs=smooth_epochs, batch_size=batch_size, validation_split=validation_split,verbose=1)

        self.smoother = smoother
        self.encoder = keras.Model(smoother.input, smoother.layers[7].output)
        print("done")

    def trainClassifier(self,X,y,X_val,y_val,class_epochs,batch_size,restarts):

        X_norm = normalizeMatrix(X)
        tics = np.log10(np.array([np.max([2, integratePeak(x)]) for x in X]))
        X_latent = self.encoder.predict(X_norm)

        X_tmp = normalizeMatrix(X_val)
        tic_val = np.log10(np.array([np.max([2, integratePeak(x)]) for x in X_val]))
        X_val = self.encoder.predict(X_tmp)

        bestLoss = np.inf
        bestWeights = -1
        bestValErr = -1
        bestBestEpoch = -1
        trainErr = -1
        trainLoss = -1

        for x in range(restarts):

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

            classifer = ClassifierLatent(X_latent.shape[1])

            classifer.fit([X_latent, tics], y, epochs=int(class_epochs),
                          batch_size=batch_size, verbose=0, callbacks=[cb, history],
                          validation_data=([X_val, tic_val], y_val))

            valLoss = history.history["val_loss"][cb.best_epoch]
            valErr = history.history["val_mean_absolute_error"][cb.best_epoch]
            bestEpoch = cb.best_epoch
            if valLoss < bestLoss:
                bestLoss = valLoss
                bestWeights = classifer.get_weights()
                bestValErr = valErr
                bestBestEpoch = bestEpoch
                trainLoss = history.history["loss"][cb.best_epoch]
                trainErr = history.history["mean_absolute_error"][cb.best_epoch]
        print("loss:",trainLoss,"mean_absolute_error:",trainErr,"val loss:", bestLoss, "val_mean_absolute_error:",bestValErr, "numEpochs:", bestBestEpoch)

        classifer.set_weights(bestWeights)

        self.classifier = classifer

        return history

    def trainClassifierActive(self,X,X_labeled,y_labeled,class_epochs,batch_size,restarts,numVal = 10,numManualPerRound=3,inJupyter=True):
        trainingInds = []

        valInds = list(range(len(X)))
        valInds = rd.sample(valInds,numVal)

        y = np.zeros((len(X)-numVal,2))

        X_val = X[valInds]
        y_val = np.zeros((numVal,2))

        X = X[[x for x in range(len(X)) if x not in valInds]]

        updatingInds = list(range(len(X)))

        for ind in range(len(X_val)):
            val = self.labelPeak([X_val[ind]], 0, self.windowSize, inJupyter,"")
            y_val[ind, 0] = 1 - val
            y_val[ind, 1] = val

        if len(X_labeled) > 0:
            self.trainClassifier(X_labeled,
                                 y_labeled,
                                 X_val,y_val, class_epochs, batch_size, restarts)

        y[updatingInds] = self.classifyMatrix(X[updatingInds])

        doMore = True
        i=0
        while doMore:
            if len(updatingInds) > 0:

                entropies = [-1 * np.sum([yyy * np.log(yyy) for yyy in yy]) for yy in y[updatingInds]]
                order = list(range(len(updatingInds)))
                order.sort(key=lambda x: entropies[x], reverse=True)
                order = [updatingInds[x] for x in order]

                if len(order) < numManualPerRound:
                    numManualPerRound = len(order)

                for ind in order[:numManualPerRound]:
                    val = self.labelPeak([X[ind]], 0, self.windowSize, inJupyter, y[ind][1])
                    y[ind, 0] = 1 - val
                    y[ind, 1] = val
                    trainingInds.append(ind)
                    updatingInds.remove(ind)

                X_train = np.concatenate((X[trainingInds],X_labeled),axis=0)
                y_train = np.concatenate((y[trainingInds],y_labeled),axis=0)

                if len(X_train) > 0:
                    self.trainClassifier(X_train,y_train,X_val,y_val,class_epochs,batch_size,restarts)

                y[updatingInds] = self.classifyMatrix(X[updatingInds])

                plt.figure()
                plt.hist(y[:, 1], bins=20)
                plt.title("round" + str(i + 1))
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


    def classifyMatrix(self,X):
        # normalize matrix
        X_norm = normalizeMatrix(X)
        peak_areas = np.log10(np.array([np.max([2, integratePeak(x)]) for x in X]))

        X_latent = self.encoder.predict(X_norm)
        y = self.classifier.predict([X_latent, peak_areas])
        return y

    def curatePeaks(self,raw_datas,peaks,threshold=0.5):
        print("generating EICs...")
        mzs = peaks["mz"].values
        rts =  peaks["rt"].values

        X = self.makeDataMatrix(raw_datas,mzs,rts)
        peak_areas = np.log10(np.array([np.max([2, integratePeak(x)]) for x in X]))

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

        for [file, index], score, intensity in zip(keys, y[:, 1], peak_areas):
            peak_scores.at[index, file] = score
            peak_intensities.at[index, file] = intensity
            val = 0
            if score > threshold:
                val = 1
            peak_curated.at[index, file] = val

        return peak_curated,peak_scores,peak_intensities

    def detectPeaks(self, rawDatas, cutoff=0.5, intensityCutoff = 100,numDataPoints=3,window=10, noiseCutoff=4):
        rois = self.roiDetection(self, rawDatas, intensityCutoff=intensityCutoff, numDataPoints=numDataPoints)

        print("generating all EICs from ROIs...")
        peaks = []
        dt = self.windowSize / self.resolution
        tmpRes = int(math.ceil((rawDatas[0].rts[-1] - rawDatas[0].rts[0] + self.windowSize) / dt))
        oldRes = int(self.resolution)
        oldwindow = float(self.windowSize)
        self.resolution = tmpRes
        self.windowSize = rawDatas[0].rts[-1] + self.windowSize/2 - rawDatas[0].rts[0] + self.windowSize/2
        rts = [rawDatas[0].rts[-1] + self.windowSize / 2 for _ in rois]
        X_tot = self.makeDataMatrix(rawDatas, rois, rts, 0)
        self.resolution = oldRes
        self.windowSize = oldwindow

        numPoints = math.floor(float(tmpRes - self.resolution) / window)

        X = np.zeros((int(numPoints * len(rois)), self.resolution))

        counter = 0

        mzs = []
        rts = []
        files = []

        start = 0
        end = self.resolution

        for rawData in rawDatas:
            rt = float(rawData.rts[0])
            for row in range(len(rois)):
                for _ in range(numPoints):
                    X[counter, :] = X_tot[row, start:end]
                    counter += 1
                    start += window
                    end += window
                    mzs.append(rois[row])
                    rts.append(rt)
                    rt += dt * window
                    files.append(rawData.filename)

        tics = np.log10(np.array([np.max([2, integratePeak(x)]) for x in X]))
        X = normalizeMatrix(X)
        print("done, ", len(X), " EICs generated")
        print("smoothing EICs...")
        X = self.encoder.predict(X, verbose=1)

        print("done")
        print("classifying peaks...")
        y = self.classifier.predict([X, tics], verbose=1)[:, 1]
        print("done")
        for mz, rt,filename, score, tic in zip(mzs, rts,files, y, tics):
            if score > cutoff and tic > noiseCutoff:
                peaks.append([mz, rt,filename, score,tic])

        if len(peaks) > 0:
            peaks = pd.DataFrame(data=np.array(peaks), columns=["mz", "rt","sample","score","peak area"])
        else:
            peaks = pd.DataFrame(columns=["mz", "rt","sample","score","peak area"])

        print(len(peaks), " peaks found")

        return peaks, X

    def label_peaks(self,raw_data,peaks,inJupyter = True):
        rt_starts = [row["rt"] - self.windowSize/2 for _,row in peaks.iterrows()]
        rt_ends = [row["rt"] + self.windowSize/2 for _,row in peaks.iterrows()]
        y = []
        mat = self.makeDataMatrix([raw_data],peaks["mz"].values,peaks['rt'])
        count = 1
        for vec,rt_start,rt_end in zip(mat,rt_starts,rt_ends):
            y.append(self.labelPeak([vec],rt_start,rt_end,inJupyter,str(count) + "/" + str(len(mat))))
            count += 1
        peaks["classification"] = y
        return peaks

    def labelPeak(self,vecs,rt_start,rt_end,inJupyter,title=""):
        plt.figure()
        xs = np.linspace(rt_start, rt_end, len(vecs[0]))
        [plt.plot(xs, vec) for vec in vecs]
        plt.xlabel("time (arbitrary)")
        plt.ylabel("intensity")
        plt.title(title)
        plt.show()
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

    def roiDetection(self,rawdatas,intensityCutoff=100,numDataPoints = 3):
        rtss = [rawdata.rts for rawdata in rawdatas]
        rois = []

        counter = 0
        totalNum = np.sum([len(rts) for rts in rtss])
        for rts,rawdata in zip(rtss,rawdatas):
            ppm = rawdata.ppm
            for rt in rts:
                printProgressBar(counter, totalNum,prefix = "Detecting ROIs",suffix=str(len(rois)) + " ROIs found",printEnd="")
                counter += 1
                for mz, i in rawdata.data[rt].items():
                    if i > intensityCutoff:
                        update,pos = binarySearchROI(rois,mz,ppm)
                        if update:
                            rois[pos]["mzs"].append(mz)
                            rois[pos]["mz_mean"] = np.mean(rois[pos]["mzs"])
                            rois[pos]["extended"] = True
                            rois[pos]["count"] += 1
                        else:
                            if pos != len(rois):
                                rois.insert(pos,{"mz_mean":mz,"mzs":[mz],"extended":True,"count":1})
                            else:
                                rois.append({"mz_mean":mz,"mzs":[mz],"extended":True,"count":1})

            toKeep = []
            for x in range(len(rois)):
                if rois[x]["extended"] == True or rois[x]["count"] >= numDataPoints:
                    toKeep.append(x)

            rois = [rois[x] for x in toKeep]

            for x in range(len(rois)):
                rois[x]["extended"] = False


        rois = [x["mz_mean"] for x in rois]
        print()
        print(len(rois)," ROIs found")

        return rois

    def filterPeakListByScores(self,peakScores,samples,cutoff,frac):
        goodInds = [index for index,row in peakScores.iterrows() if float(len([x for x in samples if row[x] > cutoff])) / len(samples) > frac]
        return peakScores.loc[goodInds,:]

    def getPeakBoundaries(self,X,samples,peakScores,cutoff):
        peakBoundaries = pd.DataFrame(index=peakScores.index.values)
        i = 0
        toFill = []
        for samp in samples:
            bounds = []
            for index,row in peakScores.iterrows():
                if row[samp] > cutoff:
                    lb,rb = findPeakBoundaries(X[i])
                    actualRts = np.linspace(row["rt"]-self.windowSize/2,row["rt"]+self.windowSize/2,self.resolution)
                    bounds.append((actualRts[lb],actualRts[rb]))
                else:
                    bounds.append((-1,-1))
                    toFill.append((index,samp))

            peakBoundaries[samp] = deepcopy(bounds)

        for index,samp in toFill:
            widths = [x[1]-x[0] for x in peakBoundaries.loc[index,:] if x[0] > 0 and x[1] > 0]
            centers = [np.mean(x) for x in peakBoundaries.loc[index,:] if x[0] > 0 and x[1] > 0]
            peakBoundaries.at[index,samp] = (np.mean(centers) - np.mean(widths)/2,np.mean(centers) + np.mean(widths)/2 )

        return peakBoundaries


def validateInput(input):
    try:
        input = float(input)
        if input not in [0,1]:
            return False
        return True
    except:
        return False

def binarySearchROI(poss, query,ppm):

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

class rawData():
    def __init__(self):
        self.data = {}
        self.type = "centroid"
        self.filename = ""

    def readRawDataFile(self,filename,ppm,type="centroid",samplename=None):
        """
         Read MS datafile

        :param filename: str, path to MS datafile
        """
        try:

            reader = mzml.read(filename.replace('"', ""))
            ms1Scans = {}
            for temp in reader:
                if temp['ms level'] == 1:
                    ms1Scans[temp["scanList"]["scan"][0]["scan start time"]] = {mz: i for mz, i in
                                                                                zip(temp["m/z array"],
                                                                                    temp["intensity array"])}
            reader.close()
            self.rts = list(ms1Scans.keys())
            self.rts.sort()
            self.data = ms1Scans
            self.type = type
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
        intensity = [np.sum([i for mz,i in self.data[rt].items() if mz > mz_start and mz < mz_end]) for rt in rts]
        return rts,intensity

    def integrateTargets(self,transitionList):
        areas = []
        for index,row in transitionList.iterrows():
            rts,intensity = self.extractEIC(row["mz"],row["rt_start"],row["rt_end"])
            area = np.trapz(intensity,rts)
            areas.append(area)
        transitionList[self.filename] = areas
        return transitionList

    def interpolate_data(self,mz,rt_start,rt_end,smoothing=1):
        rts,intensity = self.extractEIC(mz,rt_start,rt_end)
        if len(rts) > 3:
            smoothing = smoothing * len(rts) * np.max(intensity)
            s = UnivariateSpline(rts,intensity,ext=1,s=smoothing)
        else:
            s = UnivariateSpline([0,5,10,15],[0,0,0,0],ext=1,s=smoothing)
        return s

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

def Classifier(resolution):
    descriminatorInput = keras.Input(shape=(resolution,))
    ticInput = keras.Input(shape=(1,))

    kernelsize = 3
    stride = 2
    max_norm_value = 2.0

    x = layers.Reshape((resolution, 1))(descriminatorInput)
    x = layers.Conv1D(2, kernelsize, strides=stride, activation='relu', kernel_constraint=max_norm(max_norm_value),
                      kernel_initializer='he_uniform')(x)

    x = layers.Conv1D(4, kernelsize, strides=stride, activation='relu', kernel_constraint=max_norm(max_norm_value),
                      kernel_initializer='he_uniform')(x)

    x = layers.Flatten()(x)

    x = layers.Dense(20, activation="relu")(x)

    x = keras.Model(descriminatorInput, x)

    tic = keras.Model(ticInput, layers.Dense(1, activation="linear")(ticInput))

    x = layers.concatenate([x.output, tic.output], axis=1)
    x = layers.Dense(10, activation="relu")(x)
    output = layers.Dense(2, activation="softmax")(x)

    classifier = keras.Model([descriminatorInput, ticInput], output, name="discriminator")

    classifier.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(1e-4),
                          metrics=['mean_absolute_error'])

    return classifier

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
    apex = int(len(peak)/2)
    #find left bound

    foundNewApex = True

    while(foundNewApex):
        foundNewApex = False

        x = apex
        while peak[x] > peak[apex] / 2 and x > 0:
            if peak[x] > peak[apex]:
                apex = x
            x -= 1
        lb = x

        x = apex
        while peak[x] > peak[apex] / 2 and x < len(peak) - 1:
            if peak[x] > peak[apex]:
                apex = x
                foundNewApex = True
            x += 1
        rb = x

    return lb,rb

def integratePeak(peak):
    lb,rb = findPeakBoundaries(peak)
    if lb != rb:
        area = np.trapz(peak[lb:rb],np.linspace(lb,rb,rb-lb))
    else:
        area = 0
    return area

