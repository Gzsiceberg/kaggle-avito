from itertools import izip
from math import log, exp
from datetime import datetime
from csv import DictReader
from itertools import izip
import argparse, random
from util import write_dump, read_dump
import numpy as np
from sklearn.metrics import log_loss
import xgboost as xgb
import theano
import lasagne
from lasagne import layers
from lasagne.objectives import Objective
from lasagne.layers import DenseLayer
from lasagne.layers import DropoutLayer
from lasagne.layers import InputLayer
from lasagne.nonlinearities import identity,sigmoid, tanh,rectify, linear
from lasagne.nonlinearities import softmax
from lasagne.updates import nesterov_momentum,adagrad
from nolearn.lasagne import NeuralNet
from nolearn.lasagne import BatchIterator

def get_nn_model(shape):
    np.random.seed(9)
    model = NeuralNet(
        layers=[  
            ('input', layers.InputLayer),
            ('hidden1', layers.DenseLayer),
            ('hidden2', layers.DenseLayer),
            ('output', layers.DenseLayer),
            ],
        input_shape=(None,  shape[1]),
        hidden1_num_units=16,  # number of units in hidden layer
        hidden1_nonlinearity=sigmoid,
        hidden2_num_units=8,  # number of units in hidden layer
        hidden2_nonlinearity=sigmoid,
        output_nonlinearity=softmax, 
        output_num_units=2,  # target values

        # optimization method:
        update=adagrad,
        update_learning_rate=theano.shared(np.float32(0.1)),

        on_epoch_finished=[
        ],
        use_label_encoder=False,

        batch_iterator_train=BatchIterator(batch_size=500),
        regression=False,  # flag to indicate we're dealing with regression problem
        max_epochs=900,  # we want to train this many epochs
        verbose=1,
        eval_size=0.0,
        )
    return model

def read_true():
    with open("data/cv_label.out") as f:
        for line in f:
            y = int(line[0])
            yield y

def read_preds(all_preds):
    all_fh = []
    for pred in all_preds:
        all_fh.append(open(pred))
    for preds in izip(*all_fh):
        yield map(float, preds)

def logloss(p, y):
    p = max(min(p, 1. - 10e-15), 10e-15)
    return -log(p) if y == 1. else -log(1. - p)

def logit(x):
    return 1. / (1. + exp(x))

def relogit(x):
    return log((1. - x) / max(x, 10e-15))

def stack_preds(preds):
    res = 1
    res2 = 0
    res3 = 0
    ratio = [1.0/len(preds) for i in range(len(preds))]
    for p, r in zip(preds, ratio):
        res *= p
        res2 += p * r
        res3 += relogit(p)
    res = res ** (1.0/len(preds))
    res3 = logit(res3 / len(preds))
    return res2
    # return res
    # return res3
    # return (res + res2 + res3) / 3.0

def read_sample(path):
    for t, row in enumerate(DictReader(open(path))):
        yield int(row["ID"])

def get_train_data():
    all_preds = [
            "data/cv.ins5.ffm.out",
            "data/xgb5_train.out",
            "data/cv.ins_bag.ffm.out",
            "data/cv.ins5.fm.out",
            "data/cv.ins3.ffm.out",
            "data/cv.ins20.ffm.out",
            "data/xgb3_train.out",

            "data/cv.ins.ffm.out",
            "data/cv.ins2.ffm.out",
            ]
    tr_X, tr_y_true = [], []
    tr_loss, te_loss = 0.0, 0.0
    te_X, te_y_true = [], []
    avg_ctr = 0
    loss = 0
    delta = 1
    now = datetime.now()

    random.seed(9)
    for t, (y, preds) in enumerate(izip(read_true(), read_preds(all_preds))):
        avg_p = stack_preds(preds)
        avg_ctr += avg_p
        l = logloss(avg_p, y)
        if t < 5000000:
            if "nn3" in args.model:
                preds = map(lambda x : relogit(x), preds)
            elif "nn" in args.model:
                preds = map(lambda x : relogit(x) + (random.random() - 0.5) / 10, preds)
            elif "xgb" in args.model:
                preds = map(lambda x : relogit(x) + (random.random() - 0.5) / 10, preds)
            tr_X.append(preds)
            tr_y_true.append(y)
            tr_loss += l
        else:
            preds = map(lambda x : relogit(x), preds)
            te_X.append(preds)
            te_y_true.append(y)
            te_loss += l
        loss += l
        if t == delta:
            print "%s: %s loss: %s"%(t, datetime.now() - now, loss/(t + 1))
            delta *= 2
    print loss/(t + 1), avg_ctr / (t + 1)
    print "train:", len(tr_X), tr_loss/len(tr_X)
    if te_X:
        print "test:", len(te_X), te_loss/len(te_X)
    return tr_X, tr_y_true, te_X, te_y_true

def cv_method():
    tr_X, tr_y_true, te_X, te_y_true = get_train_data()
    if "nn" in args.model:
        tr_X = np.array(tr_X).astype(np.float32)
        tr_y_true = np.array(tr_y_true).astype(np.int32)
        model = get_nn_model(tr_X.shape)
        model.fit(tr_X, tr_y_true)
        write_dump("%s_model.dump"%args.model, model)
        if te_X:
            te_X = np.array(te_X).astype(np.float32)
            preds = model.predict_proba(te_X)[:, 1]
            np.savetxt("nn_preds.txt", preds)
            print log_loss(te_y_true, preds)
    elif "xgb" in args.model:
        dtrain = xgb.DMatrix(tr_X, label=tr_y_true)
        if args.predict == "cv":
            if te_X:
                dtest = xgb.DMatrix(te_X, label=te_y_true)
            param = {
                    'max_depth':3,
                    'eta':0.1,
                    'silent':1,
                    'objective':'binary:logistic',
                    "eval_metric": "logloss",
                    "nthread": 9,
                    }
            if te_X:
                watchlist  = [(dtrain,'train'), (dtest, "eval")]
            else:
                watchlist  = [(dtrain,'train'),]
            num_round = 132
            bst = xgb.train(param, dtrain, num_round, watchlist)
            bst.save_model("%s_model.dump"%args.model)
            if te_X:
                preds = bst.predict(dtest)
                np.savetxt("xgb_preds.txt", preds)

def stack_method():
    if "mean" in args.model:
        all_preds = [
            "data/te.ins5.ffm.out",
            "data/xgb5_test.out",
            "data/te.ins_bag.ffm.out",
            "data/te.ins5.fm.out",
            "data/te.ins3.ffm.out",
            "data/te.ins20.ffm.out",
            "data/xgb3_test.out",
            "data/te.ins.ffm.out",
        ]
    else:
        all_preds = [
            "data/te.ins5.ffm.out",
            "data/xgb5_test.out",
            "data/te.ins_bag.ffm.out",
            "data/te.ins5.fm.out",
            "data/te.ins3.ffm.out",
            "data/te.ins20.ffm.out",
            "data/xgb3_test.out",

            "data/te.ins.ffm.out",
            "data/te.ins2.ffm.out",
        ]
    delta = 1
    now = datetime.now()
    X = [] 
    true_y = []
    for t, preds in enumerate(read_preds(all_preds)):
        if "nn" in args.model or "xgb" in args.model:
            preds = map(lambda x : relogit(x), preds)
        X.append(preds)
        true_y.append(0)
        if t == delta:
            print "%s: %s"%(t, datetime.now() - now)
            delta *= 2
    if "nn" in args.model:
        model = read_dump("%s_model.dump"%args.model)
        X = np.array(X).astype(np.float32)
        preds = model.predict_proba(X)[:, 1]
    elif "xgb" in args.model:
        dtrain = xgb.DMatrix(np.array(X), label=true_y)
        bst = xgb.Booster({'nthread':9})
        bst.load_model("%s_model.dump"%args.model)
        preds = bst.predict(dtrain)
    elif args.model == "mean":
        X = np.array(X).astype(np.float32)
        preds = np.mean(X, 1)
    with open("result_%s.csv"%args.model, 'w') as outfile:
        outfile.write('ID,IsClick\n')
        avg_p = 0
        cnt = 0
        for ID, p in izip(read_sample("data/sampleSubmission.csv"), preds):
            avg_p += p
            cnt += 1
            outfile.write('%s,%s\n' % (ID, str(p)))
        print cnt, avg_p / cnt

def main():
    if args.predict == "pred":
        ratio = 1.0
        with open(args.output, 'w') as outfile:
            outfile.write('ID,IsClick\n')
            avg_p = 0
            cnt = 0
            for ID, preds in izip(read_sample("data/sampleSubmission.csv"), read_preds([args.input])):
                preds[0] = min(preds[0] * ratio, 1.0)
                avg_p += preds[0]
                cnt += 1
                outfile.write('%s,%s\n' % (ID, str(preds[0])))
            print avg_p / cnt
    elif args.predict == "cv":
        cv_method()
    elif args.predict == "stack":
        stack_method()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--predict', type=str, default="cv")
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--input', type=str, default=None)
    parser.add_argument('--model', type=str, default="nn")
    args = parser.parse_args()

    main()
