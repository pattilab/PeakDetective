library("xcms")
library("CAMERA")
library("IPO")

pathData <- "D:/PeakDetective/data/covid_plasma/" #path to your .mzML 


setwd(pathData)
files = list.files(pattern = "*.mzML")
numFiles = length(files)

peakpickingParameters <- getDefaultXcmsSetStartingParams('centWave')
peakpickingParameters$min_peakwidth <- c(2, 20)
peakpickingParameters$max_peakwidth <- c(20,80)
peakpickingParameters$ppm <- c(15,50)
peakpickingParameters$snthresh <- c(1,10)
peakpickingParameters$noise <- c(100,10000)
peakpickingParameters$prefilter <- c(1,5)


time.xcmsSet <- system.time({ # measuring time
  resultPeakpicking <- 
    optimizeXcmsSet(files = files, 
                    params = peakpickingParameters, 
                    nSlaves = 10, 
                    subdir = NULL,
                    plot = TRUE)
})
  

ppm = resultPeakpicking$best_settings$parameters$ppm
minwidth = resultPeakpicking$best_settings$parameters$min_peakwidth
maxWidth = resultPeakpicking$best_settings$parameters$max_peakwidth
noise = resultPeakpicking$best_settings$parameters$noise
s2n = resultPeakpicking$best_settings$parameters$snthresh
mzDiff = resultPeakpicking$best_settings$parameters$mzdiff
prefilter = resultPeakpicking$best_settings$parameters$prefilter


