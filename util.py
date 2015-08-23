from csv import DictReader
from datetime import datetime, timedelta
from collections import defaultdict
import cPickle as pickle
from math import exp, log, sqrt
import random, gc

def convert_ts(d):
    return (d - datetime(1970,1,1)).total_seconds()
    
def check_row(row):
    for k in row:
        if k == row[k]:
            return True
        return False
    return True

def read_tsv(file_path, now=datetime.now(), max_lines=None, delimiter="\t"):
    print "-" * 80
    print "reading %s"%file_path
    delta = 1
    for t, row in enumerate(DictReader(open(file_path), delimiter=delimiter)):    
        if check_row(row):
            continue
        if t == delta:
            print "%s: %s file_name: %s"%(t, datetime.now() - now, file_path)
            delta *= 2
            if max_lines and delta > max_lines:
                break
        yield t, row
        
def write_dump(file_name, obj):
    with open(file_name, "w") as wfile:
        pickle.dump(obj, wfile, protocol=pickle.HIGHEST_PROTOCOL)

def read_dump(file_name):
    with open(file_name) as rfile:
        return pickle.load(rfile)

def cache(file_name):
    def gen_cache_func(f):
        def cache_func(args_name="", *args, **kwargs):
            if args_name:
                dump_file_name = "%s_%s"%(args_name, file_name)
            else:
                dump_file_name = file_name
            try:
                res = read_dump(dump_file_name)
            except Exception, e:
                print e
                res = f(*args, **kwargs)
                write_dump(dump_file_name, res)
            return res
        cache_func.__name__ = f.__name__
        return cache_func
    return gen_cache_func

def logloss(p, y):
    p = max(min(p, 1. - 10e-15), 10e-15)
    return -log(p) if y == 1. else -log(1. - p)

def next_row(it):
    all_rows = []
    for t, row in it:
        row["SearchID"] = int(row["SearchID"])
        if not all_rows:
            all_rows.append(row)
        elif all_rows[0]["SearchID"] == row["SearchID"]:
            all_rows.append(row)
        else:
            yield all_rows[0]["SearchID"], all_rows
            all_rows = [row,]
        # if t > 1000:
        #     break
    if all_rows:
        yield all_rows[0]["SearchID"], all_rows

@cache("data/get_sid_to_uid.dump")
def get_sid_to_uid():
    uid_sids = defaultdict(list)
    for t, row in read_tsv("data/SearchInfo.tsv"):
        sid, uid = int(row["SearchID"]), int(row["UserID"])
        date_str = row["SearchDate"]
        ts = convert_ts(datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S.0"))
        l = uid_sids[uid]
        if len(l) < 5:
            uid_sids[uid].append((ts, sid))
        else:
            l.sort()
            if ts > l[0][0]:
                l[0] = (ts, sid)
    print "uid_sids size: %s"%len(uid_sids)
    for uid in uid_sids:
        uid_sids[uid].sort()
    return uid_sids

def get_category():
    par_dict = {}
    for t, row in read_tsv("data/Category.tsv"):
        cid = int(row["CategoryID"])
        pid = int(row["ParentCategoryID"])
        par_dict[cid] = pid
    return par_dict

@cache("data/get_test_ids.dump")
def get_test_ids():
    test_ids = set()
    for t, row in read_tsv("data/testSearchStream.tsv"):
        test_ids.add(int(row["SearchID"]))
    return test_ids

@cache("data/get_ids_new.dump")
def get_ids():
    uid_sids = get_sid_to_uid()
    test_ids = get_test_ids()
    print "uid_sids size: %s"%len(uid_sids)
    min_ts, max_ts = None, None
    min_test_ts, max_test_ts = None, None
    for uid in uid_sids:
        d = uid_sids[uid][-1][0]
        if min_ts is None:
            min_ts = d
        if max_ts is None:
            max_ts = d
        min_ts = min(min_ts, d)
        max_ts = max(max_ts, d)

        sz = len(uid_sids[uid])
        array = uid_sids[uid]
        for i in range(sz - 1, max(0, sz - 3), -1):
            ts = array[i][0]
            sid = array[i][1]
            if sid in test_ids:
                if min_test_ts is None:
                    min_test_ts = ts
                if max_test_ts is None:
                    max_test_ts = ts
                min_test_ts = min(min_test_ts, ts)
                max_test_ts = max(max_test_ts, ts)

    begin_ts = max_ts - timedelta(days=6).total_seconds()
    # begin_ts = min_test_ts
    end_ts = max_ts
    cv_ids = set()
    only_cv_uids = set()
    only_test_uids = set()
    print "%s %s"%(datetime.utcfromtimestamp(begin_ts), datetime.utcfromtimestamp(end_ts))
    for uid in uid_sids:
        uid_sid = uid_sids[uid]

        test_sid_cnt = 0
        for i in range(len(uid_sid)):
            ts = uid_sids[uid][-(i + 1)][0]
            sid = uid_sids[uid][-(i + 1)][1]
            if sid not in test_ids:
                break
            test_sid_cnt += 1
        if test_sid_cnt != len(uid_sid):
            if len(uid_sid) - test_sid_cnt <= 1:
                only_cv_uids.add(uid)
            if ts >= begin_ts and ts <= end_ts:
                cv_ids.add(sid)
        else:
            only_test_uids.add(uid)
    print "test_ids size: %s"%len(test_ids)
    print "cv_ids size: %s"%len(cv_ids)
    print "only_in_test: %s only_in_cv: %s"%(len(only_test_uids), len(only_cv_uids))
    return {"cv_ids": cv_ids, 
            "only_cv_uids": only_cv_uids,  
            "only_test_uids": only_test_uids,
            }

def data(test=False, 
        train_iter=next_row(read_tsv("data/trainSearchStream.tsv")), 
        test_iter=next_row(read_tsv("data/testSearchStream.tsv")), 
        sinfo_iter=read_tsv("data/SearchInfo.tsv"),
        maxlines=1e6,
        ):
    ids_map = get_ids()
    cv_ids = ids_map["cv_ids"]
    only_cv_uids = ids_map["only_cv_uids"]
    only_test_uids = ids_map["only_test_uids"]
    print "cv_ids: %s only_cv_uids: %s only_test_uids: %s"%(len(cv_ids), len(only_cv_uids), len(only_test_uids))

    tr_rows, te_rows = [], []
    tr_cnt, cv_cnt, te_cnt = 0, 0, 0
    cv_ins_cnt = 0
    cv_loss = 0
    while True:
        if tr_cnt > maxlines and test:
            print "%s %s %s"%(tr_cnt, cv_cnt, te_cnt)
            break
        if random.randint(0, 1e6) == 0 and cv_ins_cnt > 0:
            print "cv_loss: %s cv_ins_cnt: %s"%(cv_loss/cv_ins_cnt, cv_ins_cnt)

        if not tr_rows:
            tr_sid, tr_rows = next(train_iter, (None, None))
        if not te_rows:
            te_sid, te_rows = next(test_iter, (None, None))
        sinfo = next(sinfo_iter, (None, None))[1]
        if sinfo is None and tr_rows is None and te_rows is None:
            break
        sinfo["SearchID"] = int(sinfo["SearchID"])
        data_type = 0
        if sinfo["SearchID"] == tr_sid:
            if tr_sid in cv_ids:
                cv_cnt += 1
                data_type = 1
                for row in tr_rows:
                    if int(row["ObjectType"]) != 3:
                        continue
                    y = 1 if row["IsClick"] == "1" else 0
                    his_ctr = float(row["HistCTR"])
                    cv_loss += logloss(his_ctr, y)
                    cv_ins_cnt += 1
            else:
                tr_cnt += 1
                data_type = 0
            yield data_type, tr_rows, sinfo
            tr_sid = tr_rows = None
        elif sinfo["SearchID"] == te_sid:
            te_cnt += 1
            data_type = 2
            yield data_type, te_rows, sinfo
            te_sid = te_rows = None
        else:
            print "tr_rows: %s te_rows: %s"%(tr_rows, te_rows)
            raise RuntimeError("sid mismatch. sinfo: %s tr: %s te: %s"%(sinfo["SearchID"], tr_sid, te_sid))
if __name__ == '__main__':
    get_ids()