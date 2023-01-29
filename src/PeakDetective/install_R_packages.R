install.packages("BiocManager",repos="http://cran.r-project.org" )
BiocManager::install("xcms")
install.packages("dplyr",repos="http://cran.r-project.org")

install.packages("devtools",repos="http://cran.r-project.org")
devtools::install_github("e-stan/mz.unity")


print("-----------------------")
print("successful installation")