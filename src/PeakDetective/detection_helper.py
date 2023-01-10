import numpy as np
import pandas as pd
import scipy.stats as stats
import os
import uuid
from . import printProgressBar, startConcurrentTask

class PeakList():
    def __init__(self,peakList = None):
        if type(peakList) == type(None):
            self.peakList = pd.DataFrame()
        else:
            self.peakList = peakList
            
    def from_csv(self,fn):
        self.peakList = pd.read_csv(fn)
        self.sampleCols = [x for x in self.peakList.columns.values if x not in ["mz","rt_start","rt_end","isotope","adduct","peak group"]] 
        
    def to_csv(self,fn):
        self.peakList.to_csv(fn)

    def to_skyline(self,fn,polarity,moleculeListName = "XCMS peaks"):
        transitionList = pd.DataFrame(self.peakList)
        transitionList["Precursor Name"] = ["unknown " + str(index) for index, row in transitionList.iterrows()]
        transitionList["Explicit Retention Time"] = [row["rt_start"] / 2 + row["rt_end"] / 2 for index, row in
                                                     transitionList.iterrows()]
        polMapper = {"Positive": 1, "Negative": -1}
        transitionList["Precursor Charge"] = [polMapper[polarity] for index, row in transitionList.iterrows()]
        transitionList["Precursor m/z"] = [row["mz"] for index,row in transitionList.iterrows()]
        transitionList["Molecule List Name"] = [moleculeListName for _ in range(len(transitionList))]
        transitionList = transitionList[
            ["Molecule List Name", "Precursor Name", "Precursor m/z", "Precursor Charge",
             "Explicit Retention Time"]]
        transitionList.to_csv(fn)



    def readXCMSPeakList(self,filename,key=".mzML"):
        data = pd.read_csv(filename,index_col=0,sep="\t")
        data_form = {}
        self.sampleCols = [x for x in data.columns.values if key in x]
        for col in self.sampleCols:
            data[col] = data[col].fillna(0)
        for index,row in data.iterrows():
            data_form[index] = {"mz":row["mzmed"],"rt":row["rtmed"]/60,"rt_start":row["rtmin"]/60,"rt_end":row["rtmax"]/60}#,"isotope_xcms":row["isotopes"],"adduct_xcms":row["adduct"],"peak group":row["pcgroup"]}
            for col in self.sampleCols:
               data_form[index][col] = row[col]
               
        self.peakList = pd.DataFrame.from_dict(data_form,orient="index")

    def runXCMS(self, path, fn, polarity, ppm, peakWidth,noise=1000,s2n=1,prefilter = 2,mzDiff=0.0001,minFrac=0.0):
        dir = os.path.dirname(__file__)
        os.system("Rscript " + os.path.join(dir, "find_peaks.R") + " " + path + " " + polarity + " " + str(
            ppm) + " " + str(peakWidth[0]) + " " + str(peakWidth[1]) + " " + fn + " " + str(noise) + " " + str(s2n) + " " + str(prefilter) + " " + str(mzDiff) + " " + str(minFrac))
        self.readXCMSPeakList(path + fn)
        
    #def filterAdductsIsotopes(self,adductsToKeep = ["[M+H]","[M-H]"],isotopesToKeep = ["[M]"]):
    #    goodRows = []
    #    for index,row in self.peakList.iterrows():
    #        if (pd.isna(row["adduct_xcms"]) or any(x in row["adduct_xcms"] for x in adductsToKeep)) and (pd.isna(row["isotope_xcms"]) or any(x in row["isotope_xcms"] for x in isotopesToKeep)):
    #            goodRows.append(index)
    #    self.peakList = self.peakList.loc[goodRows,:]

    def removeRedundancy(self,corrThresh,rtThresh,polarity,ppm,numCores):
        groups = []
        anchors = []
        for index,row in self.peakList.iterrows():
            unique = True
            for i,inds,[rt,vec] in zip(range(len(groups)),groups,anchors):
                if np.abs(rt-row["rt"]) < rtThresh:
                    if len(self.sampleCols) > 2:
                        r,p = stats.pearsonr(vec,row[self.sampleCols].values)
                        if r > corrThresh:
                            unique = False
                            groups[i].append(index)
                            break
                    else:
                        unique = False
                        groups[i].append(index)
                        break
            if unique:
                groups.append([index])
                anchors.append([row["rt"],row[self.sampleCols].values])

        args = []
        for i,g in enumerate(groups):
            filt = self.peakList.loc[g,:]
            args.append([filt,polarity,ppm])
        result = startConcurrentTask(runMzUnity,args,numCores,"running mz.unity",len(groups))
        self.peakList["uniqueIon"] = True
        for res,g in zip(result,groups):
            self.peakList.loc[g,res.columns.values] = res.values

        self.peakList = self.peakList[self.peakList["uniqueIon"]]


        
    def backgroundSubtract(self,blank_keys,sample_keys,factor=3):
        goodRows = []
        sampleCols = [x for x in self.sampleCols if any(y in x for y  in sample_keys)]
        blankCols = [x for x in self.sampleCols if any(y in x for y  in blank_keys) and x not in sampleCols]
        for index,row in self.peakList.iterrows():
            blankInt = np.mean(row[blankCols])
            sampleInt = np.mean(row[sampleCols])
            if sampleInt >= factor * blankInt:
                goodRows.append(index)
                
        self.peakList = self.peakList.loc[goodRows,:]
        

def compareRT(feat1,feat2,rtTol):

    rt1 = np.mean([feat1["rt_start"],feat1["rt_end"]])
    rt2 = np.mean([feat2["rt_start"],feat2["rt_end"]])
    
    if np.abs(rt1-rt2) < rtTol:
        return True
        
    return False

def mergePeakLists(peakLists,names,ppm=20,rtTol=.5):
    peakLists = [x[["mz","rt_start","rt_end"]].reset_index() for x in peakLists]
    mergedList = pd.DataFrame()
    peakFounders = {n:{} for n in names}

    for x,n in zip(peakLists,names):
        for index,row in x.iterrows():
            new = True
            mz_bounds = [row["mz"] - ppm*row["mz"]/1e6,row["mz"] + ppm*row["mz"]/1e6]
            if len(mergedList) > 0:
                filt = mergedList[(mergedList["mz"] > mz_bounds[0]) & (mergedList["mz"] < mz_bounds[1])]            
                for index2,row2 in filt.iterrows():
                    if compareRT(row,row2,rtTol):
                        new = False
                        peakFounders[n][index2] = 1 
                        break
            if new:
                ind = len(mergedList)
                mergedList = pd.concat((mergedList,x.iloc[index:index+1,:]),axis=0,ignore_index=True)
                for n2 in names:
                    peakFounders[n2][ind] = 0
                peakFounders[n][ind] = 1
                
      
                            
    peakFounders = pd.DataFrame.from_dict(peakFounders,orient="index").transpose()
    mergedList = pd.concat((mergedList,peakFounders),axis=1,ignore_index=False)
    return mergedList

def runMzUnity(df,polarity,ppm,q=None):
    dir = os.path.dirname(__file__)
    path = str(uuid.uuid4()) + ".csv"
    if len(df) > 0:
        df.to_csv(path)
        os.system("Rscript " + os.path.join(dir, "mz_unity.R") + " " + path + " " + str(polarity) + " " + str(ppm) + " " + path)
        df = pd.read_csv(path,index_col=0)
        os.remove(path)

    else:
        df["uniqueIon"] = True

    if type(q) != type(None):
        q.put(0)

    return df

            
        
        
        
                
        
    
     
        