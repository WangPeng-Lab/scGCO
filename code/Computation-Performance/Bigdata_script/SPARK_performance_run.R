##-------------------------------------------------------------
##  Computer performance comparison  
##-------------------------------------------------------------
rm(list = ls())
gc()
# load the R package
library(SPARK)

tt=c()
mem=c()
#n_cells=c()
n_cell=c(10000) #,250,500) #,1000,1500,5000,10000,15000,50000,100000,500000,1000000)
for (i in 1:length(n_cell)){
ff='../../../data/Simulation_data/1M_cells_100genes_counts_0.csv'
counts= read.csv(ff,header = TRUE,row.names=1,nrows=n_cell[i])
rn <- rownames(counts)
info <- cbind.data.frame(x = as.numeric(sapply(strsplit(rn, split = "x"), "[", 1)), 
                         y = as.numeric(sapply(strsplit(rn, split = "x"), "[", 2)))
rownames(info) <- rn

spark <- CreateSPARKObject(counts = t(counts), location = info[, 1:2], 
                           percentage = 0.1, min_total_counts = 10)

spark@lib_size <- apply(spark@counts, 2, sum)
m = memory.size(max=TRUE)
print(m)

t0=Sys.time()
spark <- spark.vc(spark, covariates = NULL, lib_size = spark@lib_size, 
                  num_core = 8, verbose = F, fit.maxiter = 500)
m0 = memory.size(max=TRUE)
print(m0)

spark <- spark.test(spark, check_positive = T, verbose = F)
t1=Sys.time()

m1 = memory.size(max=TRUE)
tt=as.numeric(difftime(t1,t0,units = "secs"))

mem[i]=m1
print(n_cell)
print(tt)
print(m1)
# n_cells[i]=paste(num_cells[i],"n.cells",sep = "")


}

results=data.frame('n.cells'= n_cell,'n.genes'= 100,'time(s)'=tt,'memory(MiB)'=mem)


