library(mz.unity)
library(dplyr)

args = commandArgs(trailingOnly=TRUE)

inputPath <- args[1]
outPath <- args[4]
polarity <- as.integer(args[2])
ppm <- as.numeric(args[3])

#load data
mz = read.csv(inputPath)

#format data
cols = colnames(mz)
mz$m = mz$mz
mz$z = polarity
mz$d = 1
mz$id = seq(nrow(mz))
mz$row = seq(nrow(mz))

#gather z=2 ions
mz2 = mz
mz2$m = mz2$m * 2
mz2$z = mz2$z * 2

doubleCharged = mz.unity.search(A = mz2, 
                        B = mz2, 
                        M = M.iso, ppm = ppm, 
                        BM.limits = cbind(M.min = c(1), M.max = c(1), B.n = c(1)))


doubleCharged = reindex(doubleCharged, '^A$|^A.|^B$|^B.', mz2$row)

doubleCharged = unique(c(doubleCharged$A,doubleCharged$B.1))

#filter out single charged
singleCharged = mz[which(!mz$row %in% doubleCharged),]

if (nrow(singleCharged) > 1){
  #find M+1 ions
  m1Ions = mz.unity.search(A = singleCharged,
                         B = singleCharged,
                         M = M.iso, ppm = ppm,
                         BM.limits = cbind(M.min = c(1), M.max = c(1), B.n = c(1)))

  m1Ions = reindex(m1Ions,'^A$|^A.|^B$|^B.', singleCharged$row)

  m1Ions = unique(m1Ions$B.1)

  #filterOutM1Ions
  mIons = singleCharged[which(singleCharged$row %in% m1Ions),]

  if (nrow(mIons) > 1){
     #find neutral loses
    nls = mz.unity.search(A = mIons,
                          B = mIons,
                          M = M.n, ppm = ppm,
                          BM.limits = cbind(M.min = c(1), M.max = c(1), B.n = c(1))
    )
    nls = reindex(nls,'^A$|^A.|^B$|^B.', mIons$row)
    nls = unique(nls$A)

    #filter nuetral lose
    uniqueIons = mIons[which(!mIons$row %in% nls),]
  }else{
    uniqueIons = mIons
  }


}else{
  uniqueIons = singleCharged
}



mz$uniqueIon = mz$row %in% uniqueIons$row


#filter cols
mz = mz[,c(cols,"uniqueIon")]

write.csv(mz,outPath,row.names=FALSE)









