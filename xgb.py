#!/usr/bin/python
import numpy as np
import scipy.sparse as sparse
import xgboost as xgb
from math import log
import argparse, sys
from sklearn.metrics import log_loss

def hash_val(t, v, dtype=None, D=22):
    return (t << D) | (hash(unicode(v)) & ((1 << D) - 1))

def evalerror(preds, dtrain):
    labels = dtrain.get_label()
    preds = preds / (preds + (1.0 - preds) / 0.1) 
    return 'loss', log_loss(np.array(labels), preds)

def main():
    tr_path = args.tr
    te_path = args.te
    dtrain = xgb.DMatrix(tr_path)
    dtrain.save_binary("%s.buffer"%tr_path)
    dtest = xgb.DMatrix(te_path)
    dtest.save_binary('%s.buffer'%te_path)

    # num_round: 1000 cv: 0.041692
    param = {'max_depth':16, 
            'eta':0.15, 
            'silent':1, 
            'objective':'binary:logistic',
            "eval_metric": "logloss", 
            "nthread": 20, 
            'min_child_weight': 10, 
            "colsample_bytree": 0.3}

    # specify validations set to watch performance
    watchlist  = [(dtest,'eval'),]
    num_round = args.round
    if args.model_in:
        bst = xgb.Booster({'nthread':20}) #init model
        bst.load_model(args.model_in) # load data
    else:
        bst = xgb.train(param, dtrain, num_round, watchlist, feval=evalerror)

    preds = bst.predict(dtest)
    preds = preds / (preds + (1.0 - preds) / 0.1)
    labels = dtest.get_label()
    print "loss: %s"%(log_loss(np.array(labels), preds))
    bst.save_model(args.model_out)
    if args.prob:
        np.savetxt(args.prob, preds)

    if args.leaf_tr_out:
        leaf_index = bst.predict(dtrain, pred_leaf=True)
        print "writing gbdt_tr..."
        file_name = args.leaf_tr_out
        np.savetxt(file_name, leaf_index, fmt="%d", delimiter=',')

    if args.leaf_te_out:
        leaf_index = bst.predict(dtest, pred_leaf=True)
        print "writing gbdt_cv..."
        file_name = args.leaf_te_out
        np.savetxt(file_name, leaf_index, fmt="%d", delimiter=',')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--tr', type=str, default=None)
    parser.add_argument('--te', type=str, default=None)
    parser.add_argument('--model_in', type=str, default=None)
    parser.add_argument('--model_out', type=str, default="0001.model")
    parser.add_argument('--prob', type=str, default=None)
    parser.add_argument('--round', type=int, default=100)
    parser.add_argument('--leaf_tr_out', type=str, default=None)
    parser.add_argument('--leaf_te_out', type=str, default=None)
    args = parser.parse_args()
    main()
