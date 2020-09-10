## Analyze with SPARK
##-----------------
rm(list = ls())
library(SPARK)

info <- read.csv("../processed_data/Rep11_MOB_info_scgco.csv", row.names = 1)
exp_diff=1

for (noise in c(0.3,0.4)) {
    for (irep in 0:9) {
    countdata <- t(read.csv(paste0("../processed_data/sim_MOB_expdiff", exp_diff, "_noise", noise,
                                   "_counts",irep,".csv"), row.names = 1))
    spark <- CreateSPARKObject(counts = countdata, location = info[, 1:2], 
                               percentage = 0.1, min_total_counts = 0)

    spark@lib_size <- apply(spark@counts, 2, sum)

    spark <- spark.vc(spark, covariates = NULL, lib_size = spark@lib_size, 
                      num_core = 1, verbose = T, fit.maxiter = 500)
    spark <- spark.test(spark, check_positive = T, verbose = T)

    write.csv(spark@res_mtest,file=paste0("../spark_results/sim_MOB_expdiff", exp_diff, "_noise", noise,"_counts",irep,
                                   "_spark.csv"),row.names = T)
        }
    }
