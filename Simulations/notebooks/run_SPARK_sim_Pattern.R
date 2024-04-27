## Analyze with SPARK
##-----------------
rm(list = ls())
library(SPARK)

pattern = "hotpot"
exp_diff=1

for (noise in c(0.1,0.2,0.3,0.4,0.5,0.6,0.7)) {
    for (irep in c(0)) {
    countdata <- read.csv(paste0("../processed_data/sim_",pattern,"_expdiff", exp_diff, "_noise", noise,
                                   "_counts",irep,".csv"), row.names = 1)
                                   
    rn <- rownames(countdata)
    info <- cbind.data.frame(x = as.numeric(sapply(strsplit(rn, split = "x"), "[", 1)), 
                     y = as.numeric(sapply(strsplit(rn, split = "x"), "[", 2)))
    rownames(info) <- rn
    spark <- CreateSPARKObject(counts = t(countdata), location = info[, 1:2], 
                               percentage = 0.1, min_total_counts = 0)

    spark@lib_size <- apply(spark@counts, 2, sum)

    spark <- spark.vc(spark, covariates = NULL, lib_size = spark@lib_size, 
                      num_core = 60, verbose = T, fit.maxiter = 500)
    spark <- spark.test(spark, check_positive = T, verbose = T)

    write.csv(spark@res_mtest,file=paste0("../spark_results/sim_",pattern,"_expdiff", exp_diff, "_noise", noise,"_counts",irep,
                                   "_spark.csv"),row.names = T)
        }
    }