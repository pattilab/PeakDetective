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
        self.encoder = keras.Model(self.classifier.input, self.classifier.layers[6].output)

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

    def makeDataMatrix(self,rawdatas,mzs,rtstarts,rtends,smoothing=0):
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

    def generateNoisePeaks(self,X_norm):
        X_noise = X_norm.flatten()
        np.random.shuffle(X_noise)
        X_noise = np.reshape(X_noise, X_norm.shape)
        X_noise = normalizeMatrix(X_noise)
        #noise_tics = np.array(tics)
        #np.random.shuffle(noise_tics)
        return X_noise#,noise_tics

    def generateGaussianPeaks(self,X_norm, centerDist, numPeaksDist=(0,2), widthFactor=0.1, heightFactor=1):
        X_signal = np.zeros(X_norm.shape) #self.generateNoisePeaks(X_norm, tics)
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

    def generateFalsePeaks(self,X_norm,tics, centerDist, numPeaksDist=(0,2), widthFactor=0.1, s2nFactor=.3,heightFactor=1):
        X_noise = self.generateNoisePeaks(X_norm)
        X_noise_peaks = self.generateGaussianPeaks(X_norm, centerDist, numPeaksDist, widthFactor, heightFactor)

        s2n = s2nFactor * np.random.random(len(X_noise_peaks))
        s2nInv = 1 - s2n
        signal_tics = np.array(tics)
        np.random.shuffle(signal_tics)

        X_noise = s2n[:, np.newaxis] * X_noise
        X_signal = s2nInv[:, np.newaxis] * X_noise_peaks

        X_signal = X_noise + X_signal
        X_signal = normalizeMatrix(X_signal)

        return X_signal, signal_tics

    def generateSignalPeaks(self,X_norm,tics,centerDist,numPeaksDist = (0,2),widthFactor = 0.1,s2nFactor=.3,heightFactor = 1):
        X_noise = self.generateNoisePeaks(X_norm)
        X_noise_peaks = self.generateGaussianPeaks(X_norm,centerDist,numPeaksDist,widthFactor,heightFactor)
        X_signal_peaks = self.generateGaussianPeaks(X_norm,[[0.45,0.5],[0.5,0.55]],(1,1),widthFactor,heightFactor)
        X_signal = X_noise_peaks + X_signal_peaks
        X_signal = normalizeMatrix(X_signal)

        s2n = s2nFactor * np.random.random(len(X_signal))
        s2nInv = 1 - s2n
        signal_tics = np.array(tics)
        np.random.shuffle(signal_tics)

        X_noise = s2n[:, np.newaxis] * X_noise
        X_signal = s2nInv[:, np.newaxis] * X_signal

        X_signal = X_noise + X_signal
        X_signal = normalizeMatrix(X_signal)

        return X_signal,signal_tics


    def curatePeaks(self,raw_datas,peaks,smooth_epochs = 10,class_epochs = 10,class_round=5,batch_size=64,validation_split=0.1,useActive = True,numManualPerRound = 3,min_peaks = 100000,shift = .5,threshold=.5,noise=1000,inJupyter = True):

        #generate data matrix
        print("generated EICs...")
        mzs = list(peaks["mz"].values)
        rt_starts = [row["rt"] - self.windowSize/2 for _,row in peaks.iterrows()]
        rt_ends = [row["rt"] + self.windowSize/2 for _,row in peaks.iterrows()]
        realEnd = len(mzs)


        if len(peaks) * len(raw_datas) < min_peaks:
            numToGet = int((min_peaks - len(peaks) * len(raw_datas))/len(raw_datas))
            print("insufficent # of EICs, shifting",numToGet,"random peaks", shift, "to", shift + 1, "minutes")
            tmp = list(range(len(mzs)))
            tmp = rd.choices(tmp,k=numToGet)

            mzs += [mzs[x] for x in tmp]
            shifts = [shift + np.random.random() for x in tmp]
            rt_starts += [rt_starts[x] + shift for x,shift in zip(tmp,shifts)]
            rt_ends += [rt_ends[x] + shift for x,shift in zip(tmp,shifts)]

        X = self.makeDataMatrix(raw_datas,mzs,rt_starts,rt_ends)

        #normalize matrix
        X_norm = normalizeMatrix(X)
        tics = np.log10(np.array([np.max([2, np.sum(x)]) for x in X]))

        X_norm_with_noise = normalizeMatrix(X + np.random.normal(noise,noise/2,X.shape))

        print("done")

        #fit autoencoder
        print("fitting smoother...")
        smoother = Smoother(self.resolution)
        smoother.fit(X_norm, X_norm, epochs=smooth_epochs, batch_size=batch_size, validation_split=validation_split,verbose=1,sample_weight=tics)
        #smoother.fit(X, X, epochs=smooth_epochs, batch_size=batch_size, validation_split=validation_split)
        self.smoother = smoother
        self.encoder = keras.Model(smoother.input, smoother.layers[6].output)
        print("done")

        indsToKeep = []
        start = 0
        for _ in raw_datas:
            indsToKeep += list(range(start,start + realEnd))
            start += len(rt_ends)

        X_norm = X_norm[indsToKeep]

        #generate synthetic data
        print("generating synthetic data...",end="")
        X_signal,signal_tics = self.generateSignalPeaks(X_norm,tics,[[0,.40],[.6,1.0]])
        X_noise,noise_tics = self.generateFalsePeaks(X_norm,tics, [[0,.40],[.6,1.0]])
        #X_noise,noise_tics = self.generateNoisePeaks(X_norm,tics)

        #create merged matrices
        y = np.array([[.5,.5] for _ in X_norm] + [[1,0] for _ in X_noise] + [[0,1] for _ in X_signal])
        X_merge = np.concatenate((X_norm,X_noise,X_signal),axis=0)
        tic_merge = np.concatenate((tics, noise_tics,signal_tics))
        #tic_merge = np.ones(tic_merge.shape)

        #smooth data
        #X_merge = self.encoder.predict(X_merge)
        X_merge = self.smoother.predict(X_merge)
        X_merge = normalizeMatrix(X_merge)

        #record real and synthethic observations
        synInds = [x for x in range(len(y)) if y[x][1] < .25 or y[x][1] > .75]
        realInds = [x for x in range(len(y)) if y[x][1] > .25 and y[x][1] < .75]
        print("done")

        print("classifying peaks...")
        updatingInds = list(realInds)
        trainingInds = list(synInds)
        numRemaining = []
        for i in range(class_round):
            print("round " + str(i+1) + ": " + str(len(updatingInds)) + " unclassified features")
            numRemaining.append(len(updatingInds))
            classifer = Classifier(self.resolution)
            #classifer = Classifier(self.resolution)

            classifer.fit([X_merge[trainingInds], tic_merge[trainingInds]], y[trainingInds], epochs=class_epochs,
                              batch_size=batch_size, validation_split=validation_split,verbose=0)
            scores = classifer.predict([X_merge[updatingInds], tic_merge[updatingInds]])
            for ind, s in zip(updatingInds, scores):
                if s[1] < 0.05 or s[1] > 0.95:
                    y[ind] = classify(s.reshape(1, -1))[0]

            if useActive:
                entropies = [-1 * np.sum([yyy * np.log(yyy) for yyy in yy]) for yy in scores]
                order = list(range(len(updatingInds)))
                order.sort(key=lambda x: entropies[x],reverse=True)
                order = [updatingInds[x] for x in order]

                if len(order) < numManualPerRound:
                    numManualPerRound = len(order)
                for ind in order[:numManualPerRound]:
                    plt.figure()
                    plt.plot(range(len(X[ind])),X[ind],label="raw data")
                    plt.plot(range(len(X[ind])),np.sum(X[ind])*X_merge[ind],label="smoothed data")
                    plt.legend()
                    plt.xlabel("time (arbitrary)")
                    plt.ylabel("intensity")
                    plt.title(classifer.predict([X_merge[ind:ind+1],tic_merge[ind:ind+1]])[0][1])
                    plt.show()
                    print("Enter classification (1=True Peak, 0=Artifact): ")
                    val = float(input())
                    while not validateInput(val):
                        print("invalid classification: ")
                        val = float(input())
                    y[ind,0] = 1-val
                    y[ind,1] = val
                    plt.close()
                    if inJupyter:
                        IPython.display.clear_output(wait=True)
                trainingInds = [x for x in range(len(y)) if y[x][1] < .25 or y[x][1] > .75 or x in order[:numManualPerRound]]
                updatingInds = [x for x in range(len(y)) if y[x][1] > .25 and y[x][1] < .75 and x not in order[:numManualPerRound]]
            else:
                trainingInds = [x for x in range(len(y)) if y[x][1] < .25 or y[x][1] > .75]
                updatingInds = [x for x in range(len(y)) if y[x][1] > .25 and y[x][1] < .75]



        scores = classifer.predict([X_merge[updatingInds], tic_merge[updatingInds]])
        y[updatingInds] = classify(scores)
        print("done")

        print("training classifier...")
        classifer = Classifier(self.resolution)
        classifer.fit([X_merge, tic_merge], y, epochs=class_epochs,
                                  batch_size=batch_size, validation_split=validation_split,verbose=1)
        self.classifier = classifer
        print("done")

        print("formatting output...",end="")

        X = X_merge[realInds]
        tics = tic_merge[realInds]
        y = y[realInds][:,1]

        scores = classifer.predict([X_merge[realInds], tic_merge[realInds]])[:,1]

        peaks_curated = {raw.filename: [] for raw in raw_datas}
        peak_scores = deepcopy(peaks)

        keys = []
        for raw in raw_datas:
            peak_scores[raw.filename] = np.zeros(len(peak_scores.index.values))
            for index in peaks.index.values:
                keys.append([raw.filename, index])

        for [file, index], score in zip(keys, scores):
            peak_scores.at[index,file] = score
            if score > threshold:
                peaks_curated[file].append(index)
        peaks_curated = {file: peaks.loc[peaks_curated[file], :] for file in peaks_curated}

        return peaks_curated,X,X_norm,tics,scores,numRemaining,peak_scores

    def label_peaks(self,raw_data,peaks,inJupyter = True):
        rt_starts = [row["rt"] - self.windowSize/2 for _,row in peaks.iterrows()]
        rt_ends = [row["rt"] + self.windowSize/2 for _,row in peaks.iterrows()]
        y = []
        mat = self.makeDataMatrix([raw_data],peaks["mz"].values,rt_starts,rt_ends)
        count = 1
        for vec in mat:
            plt.figure()
            plt.plot(range(len(vec)), vec)
            plt.xlabel("time (arbitrary)")
            plt.ylabel("intensity")
            plt.title(str(count) + "/" + str(len(mat)))
            plt.show()
            print("Enter classification (1=True Peak, 0=Artifact): ")
            val = float(input())
            while not validateInput(val):
                print("invalid classification: ")
                val = float(input())

            y.append(val)
            plt.close()
            if inJupyter:
                IPython.display.clear_output(wait=True)
            count += 1
        peaks["classification"] = y
        return peaks

    def roiDetection(self,rawdata,intensityCutuff=100,numDataPoints = 3):
        rts = rawdata.rts
        rois = [{"mz_mean":mz,"mzs":[mz],"extended":False,"count":1} for mz, i in rawdata.data[rts[0]].items() if i > intensityCutuff]
        rois.sort(key=lambda x:x["mz_mean"])

        def binarySearchROI(poss,query):

            pos = bisect_left([x["mz_mean"] for x in poss],query)

            if pos == len(poss):
                if 1e6 * np.abs(query-poss[-1]["mz_mean"]) / query < rawdata.ppm:
                    return True, pos - 1
                else:
                    return False, pos
            elif pos == 0:
                if 1e6 * np.abs(query-poss[0]["mz_mean"]) / query < rawdata.ppm:
                    return True, 0
                else:
                    return False, pos
            else:
                ldiff = 1e6 * np.abs(query - poss[pos-1]["mz_mean"]) / query
                rdiff = 1e6 * np.abs(query - poss[pos]["mz_mean"]) / query

                if ldiff < rdiff:
                    if ldiff < rawdata.ppm:
                        return True, pos - 1
                    else:
                        return False, pos
                else:
                    if rdiff < rawdata.ppm:
                        return True, pos
                    else:
                        return False, pos


        counter = 0
        for rt in rts[1:]:
            printProgressBar(counter, len(rts),prefix = "Detecting ROIs",suffix=str(len(rois)) + " ROIs found",printEnd="")
            counter += 1
            for mz, i in rawdata.data[rt].items():
                if i > intensityCutuff:
                    update,pos = binarySearchROI(rois,mz)
                    #print(update,len(rois),pos)
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
                #if not(all(rois[i]["mz_mean"] <= rois[i+1]["mz_mean"] for i in range(len(rois) - 1))):
                #   print("error")



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

    def plot_classifier_interpretation(self,numPer = 100):
        sal_image = np.zeros((100, self.resolution))
        for col in range(self.resolution):
            printProgressBar(col, self.resolution,prefix = "Building classification map",printEnd="")
            for row in range(100):
                X = np.random.random((numPer, self.resolution))
                X[:, col] = float(row) / 100 * np.ones(X[:, col].shape)
                tics = np.ones(numPer)
                otherCols = [x for x in range(X.shape[1]) if x != col]
                X[:, otherCols] = (1 - float(row) / 100) * X[:, otherCols] / X[:, otherCols].sum(axis=1)[:, np.newaxis]
                scores = self.classifier([X, tics])[:, 1]
                sal_image[row, col] = np.mean(scores)
        plt.imshow(sal_image)
        plt.xlabel("time")
        plt.ylabel("relative intensity")
        plt.colorbar(label="mean output score")
        return sal_image

    def detectPeaks(self,rawData,rois,cutoff=0.5,window=10,time=1.0,noiseCutoff=4):
        print("generating all EICs from ROIs...")
        peaks = []
        dt = time/self.resolution
        tmpRes = int(math.ceil((rawData.rts[-1] - rawData.rts[0] + time) / dt))
        rt_starts = [rawData.rts[0]-time/2 for _ in rois]
        rt_ends = [rawData.rts[-1]+time/2 for _ in rois]
        oldRes = int(self.resolution)
        self.resolution = tmpRes
        X_tot = self.makeDataMatrix([rawData],rois,rt_starts,rt_ends,0)
        self.resolution = oldRes

        numPoints = math.floor(float(tmpRes - self.resolution)/window)


        X = np.zeros((int(numPoints * len(rois)),self.resolution))

        counter = 0

        mzs = []
        rts = []

        for row in range(len(rois)):
            start = 0
            end = self.resolution
            rt = float(rawData.rts[0])
            for _ in range(numPoints):
                X[counter,:] = X_tot[row,start:end]
                counter += 1
                start += window
                end += window
                mzs.append(rois[row])
                rts.append(rt)
                rt += dt * window


        ticsOrig = np.log10(np.array([np.sum(x) for x in X]))
        tics = ticsOrig#np.ones(ticsOrig.shape)
        X = normalizeMatrix(X)
        print("done, ",len(X)," EICs generated")
        print("smoothing EICs...")
        X = self.smoother.predict(X,verbose=1)
        print("done")
        print("classifying peaks...")
        y = self.classifier.predict([X,tics],verbose=1)[:,1]
        print("done")
        for mz,rt,score,tic in zip(mzs,rts,y,ticsOrig):
            if score > cutoff and tic > noiseCutoff:
                peaks.append([mz,rt,score])

        peaks = pd.DataFrame(data=np.array(peaks), columns=["mz", "rt","score"])
        print(len(peaks)," peaks found")

        return peaks




def validateInput(input):
    if input not in [0,1]:
        return False
    return True

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

    x = layers.Conv1D(16, kernelsize, strides=stride, activation='relu', kernel_constraint=max_norm(max_norm_value),
                     kernel_initializer='he_uniform')(x)

    x = layers.Conv1D(8, kernelsize, strides=stride, activation='relu', kernel_constraint=max_norm(max_norm_value),
                      kernel_initializer='he_uniform')(x)

    x = layers.Conv1D(4, kernelsize, strides=stride, activation='relu', kernel_constraint=max_norm(max_norm_value),
                      kernel_initializer='he_uniform')(x)

    x = layers.Flatten()(x)

    x = layers.Dense(5, activation="relu")(x)

    x = layers.Dense(int((resolution-8) * 4), activation="relu")(x)

    x = layers.Reshape((resolution-8, 4))(x)

    x = layers.Conv1DTranspose(8, kernelsize, strides=stride, activation='relu',
                               kernel_constraint=max_norm(max_norm_value), kernel_initializer='he_uniform')(x)

    x = layers.Conv1DTranspose(16, kernelsize, strides=stride, activation='relu',
                              kernel_constraint=max_norm(max_norm_value), kernel_initializer='he_uniform')(x)

    x = layers.Conv1DTranspose(32, kernelsize, strides=stride, activation='relu',
                              kernel_constraint=max_norm(max_norm_value), kernel_initializer='he_uniform')(x)

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
    x = layers.Conv1D(8, kernelsize, strides=stride, activation='relu', kernel_constraint=max_norm(max_norm_value),
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

    x = layers.Dense(resolution, activation="relu")(descriminatorInput)
    x = keras.Model(descriminatorInput, x)

    tic = keras.Model(ticInput, layers.Dense(1, activation="linear")(ticInput))

    x = layers.concatenate([x.output, tic.output], axis=1)

    x = layers.Dense(3, activation="relu")(x)
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