import sys
import numpy as np
from scipy import stats
import pandas as pd

for fpath in sys.argv[1:]:
    df = pd.read_csv(fpath, sep='\t')
    attributes = np.unique(df.attribute)
    targets = np.unique(df.target)

    for attr in attributes:
        tmp = df[df.attribute == attr]
        targ1 = tmp[tmp.target == targets[0]].log_probs
        targ2 = tmp[tmp.target == targets[1]].log_probs

        targ1_mean = np.mean(targ1)
        targ2_mean = np.mean(targ2)

        wilcoxon = stats.wilcoxon(targ1, targ2)

        print('****', attr, '****')
        print(f'{targets[0]} mean,\t', targ1_mean)
        print(f'{targets[1]} mean,\t', targ2_mean)

        print("Test statistic,\t", wilcoxon[0])
        print("p-value,\t", wilcoxon[1])
        print()
