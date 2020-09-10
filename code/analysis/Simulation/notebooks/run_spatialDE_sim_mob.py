import pandas as pd
import numpy as np
import NaiveDE
import SpatialDE
import time


info=pd.read_csv("../processed_data/Rep11_MOB_info_scgco.csv",index_col=0)
exp_diff=1


for noise in [0.1,0.2,0.3,0.4,0.5,0.6]:
    for irep in range(10):
        ff="../processed_data/sim_MOB_expdiff"+str(exp_diff)+"_noise"+str(noise)+"_counts"+str(irep)+".csv"
        print(ff)

        df = pd.read_csv(ff, index_col=0)
        df = df.T[df.sum(0) >= 3].T  
        sample_info = info.copy()


        X = sample_info[['x', 'y']]

        start_time = time.time()
        dfm = NaiveDE.stabilize(df.T).T
        res = NaiveDE.regress_out(sample_info, dfm.T, 'np.log(total_counts)').T
        res['log_total_count'] = np.log(sample_info['total_counts'])
        results = SpatialDE.run(X, res)

        ff="../spatialde_results/sim_MOB_expdiff"+str(exp_diff)+"_noise"+str(noise)+"_counts"+str(irep)+"_spe.csv"
        results.to_csv(ff)
