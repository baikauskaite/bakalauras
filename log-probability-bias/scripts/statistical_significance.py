import sys
import numpy as np
from scipy import stats
import pandas as pd

def get_effect_size(df1, df2, k="log_probs"):
    diff = (df1[k].mean() - df2[k].mean())
    std_ = pd.concat([df1, df2], axis=0)[k].std() + 1e-8
    return diff / std_

def exact_mc_perm_test(xs, ys, nmc=100000):
    n, k = len(xs), 0
    diff = np.abs(np.mean(xs) - np.mean(ys))
    zs = np.concatenate([xs, ys])
    for j in range(nmc):
        np.random.shuffle(zs)
        k += diff < np.abs(np.mean(zs[:n]) - np.mean(zs[n:]))
    return k / nmc

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

    attr1 = df[df.attribute == attributes[0]]
    attr2 = df[df.attribute == attributes[1]]
    print("Effect size,\t", get_effect_size(attr1, attr2))
    print("Permutation test p-value,\t", exact_mc_perm_test(attr1.log_probs, attr2.log_probs))
