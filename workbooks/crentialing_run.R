# installation

install.packages("devtools")
devtools::install_github("pattilab/credential", ref="release/3.1")

# load required library
library(credential)
library(xcms)
library(data.table)

# Step 1 Set the working directory.
setwd("D:/PeakDetective/data/cred_RP_T3/") 

# Step 2 Pre-processing LC/MS data 
xs_1 = xcmsSet(c("1T1","1T2"), method="centWave", ppm=25, peakwidth=c(5,60), snthresh=5, prefilter=c(3,1000), mzdiff=0.001)
# 1st round retention time alignment
xs_1 = retcor(xs_1, method="obiwarp",profStep=1)
# 1st round feature grouping
xs_1 = group(xs_1, bw=5, mzwid=.015, minfrac=1)
# 2nd round retention time alignment
xs_1 = retcor(xs_1, method="obiwarp",profStep=1)
# 2nd round feature grouping
xs_1 = group(xs_1, bw=5, mzwid=.015, minfrac=1)
# fillPeaks
xs_1 = fillPeaks(xs_1)

# if export=T, feature tables and a histogram summary of retention time shift, ppm error, peakwidth will be exported.
features <- credential::getXCMSfeature(xs = xs_1, intchoice="into", sampling = 1, sampleclass = NULL, export = T)
feature1t1 <- features$`1T1-credTable`
feature1t2 <- features$`1T2-credTable`


# automatic credentialing 
credentialed_peaks <- credential::credentialing(peaktable1 = feature1t1, peaktable2 = feature1t2, ppm = 15, rtwin = 5, rtcom =10, ratio1 = 1/1, ratio2 = 1/2, 
                                             ratio_tol = 0.1, ratio_ratio_tol = 0.5, cd = 13.00335-12, charges = 1:2, mpc = c(12,120), maxnmer = 4,
                                             export = T, plot = T, projectName = "credential_demo")
