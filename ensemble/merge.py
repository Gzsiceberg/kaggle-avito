from itertools import izip
from math import log, exp
from datetime import datetime
from csv import DictReader
from itertools import izip
import argparse

def read_preds(all_preds):
    all_fh = []
    for pred in all_preds:
        all_fh.append(DictReader(open(pred)))
    for preds in izip(*all_fh):
        yield preds

now = datetime.now()

all_preds = [
            "result_nn.csv",
            "result_xgb.csv",
            ]

def logit(x):
    return 1. / (1. + exp(-x))

def relogit(x):
    return -log(((1. - x) / x))

with open("avg.csv", 'w') as outfile:
    outfile.write('ID,IsClick\n')
    avg_ctr = 0
    ratio = [0.6, 0.4]
    for t, preds in enumerate(read_preds(all_preds)):
        ID = preds[0]["ID"] 
        ctr = 0.0
        for p, r in zip(preds, ratio):
            ctr += r * float(p["IsClick"])
        avg_ctr += ctr
        outfile.write('%s,%s\n' % (ID, ctr))
    print "%s"%(avg_ctr/(t + 1))
