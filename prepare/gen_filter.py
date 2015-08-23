from csv import DictReader
from datetime import datetime, timedelta
from collections import defaultdict
import cPickle as pickle
from math import exp, log, sqrt
import random, gc
from util import read_dump, write_dump, cache, read_tsv, convert_ts, data, next_row, get_category
import argparse, ast, re, json

def filter_row(row, data_type, sr):
    object_type = int(row["ObjectType"])
    if object_type != 3:
        return False
    y = int(row.get("IsClick", 0))
    if data_type == 0 and y == 0 and random.random() > sr:
        return False
    return True

def trans_ad_info(ad_info):
    if int(ad_info["IsContext"]) == 0:
        return None
    trans_keys = [
                "CategoryID", 
                "Price",
                "Params",
                "Title",
                ]
    del_keys = ["AdID", "IsContext", "_id", "LocationID",] 
    for k in del_keys:
        if k in ad_info:
            del ad_info[k]

    for key in trans_keys:
        val = ad_info[key]
        if key == "Price":
            if val == "":
                pass
            else:
                ad_info[key] = float(ad_info[key])
        elif key == "Params":
            params = ad_info[key]
            params = ast.literal_eval(params) if params else {}
            for par_key in params:
                params[par_key] = unicode(params[par_key], "utf-8")
            val = tuple([hash_val(0, (k, v)) for k, v in params.items()])
            if len(val) == 0:
                val = (-1,)
            ad_info[key] = val
        elif key == "Title":
            if not isinstance(ad_info[key], unicode):
                ad_info[key] = unicode(ad_info[key], "utf-8")
        else:
            if val == "":
                val = -1
            ad_info[key] = int(val)
    return ad_info

ad_info_list = []
ad_info_iter = read_tsv("data/AdsInfo.tsv")
def get_ad_info(aid):
    while aid - 1 >= len(ad_info_list):
        t, row = next(ad_info_iter, (None, None))
        if row is None:
            break
        ad_info_list.append(trans_ad_info(row))
    return ad_info_list[aid - 1]

def hash_val(t, v, dtype=None, D=22):
    if dtype == "xgb":
        return u"%s:%s"%(t, v)
    else:
        return (t << D) | (hash(unicode(v)) & ((1 << D) - 1))

se_params_iter = read_tsv("data/search_params.csv", delimiter=",")
se_param_list = [None]
def get_se_param(sid):
    while se_param_list[0] is None or se_param_list[0]["SearchID"] < sid:
        t, se_param = next(se_params_iter, (None, None))
        se_param["SearchID"] = int(se_param["SearchID"])
        params = json.loads(se_param["SearchParams"])
        se_param["SearchParams"] = [hash_val(0, (int(k), v)) for (k, v) in params.items()]
        se_param_list[0] = se_param 
    params = [-1,] if se_param_list[0]["SearchID"] != sid else se_param_list[0]["SearchParams"]
    return params

def main():
    random.seed(args.seed)
    data_iter = data(args.test, maxlines=args.maxl)
    print "sr: %s"%args.sr

    uid_cnt = defaultdict(int)
    ipid_cnt = defaultdict(int)
    adid_cnt = defaultdict(int)
    query_cnt = defaultdict(int)
    title_cnt = defaultdict(int)
    query_param_cnt = defaultdict(int)
    ad_param_cnt = defaultdict(int)

    for line_cnt, (data_type, rows, sinfo) in enumerate(data_iter):
        rows = filter(lambda x: filter_row(x, data_type, sr=args.sr), rows)
        if not rows:
            continue
        ipid, uid = map(int, (sinfo["IPID"], sinfo["UserID"]))
        uid_cnt[uid] += len(rows)
        ipid_cnt[ipid] += len(rows)

        query = unicode(sinfo["SearchQuery"], "utf-8")
        val = map(lambda x : hash_val(0, x), query.split())
        for v in val:
            query_cnt[v] += len(rows)

        sid = int(sinfo["SearchID"])
        for v in get_se_param(sid):
            query_param_cnt[v] += len(rows)

        for row in rows:
            aid = int(row["AdID"])
            adid_cnt[aid] += 1

            ad_info = get_ad_info(aid)
            for v in ad_info["Params"]:
                ad_param_cnt[v] += 1

            title = ad_info["Title"]
            title_val = map(lambda x : hash_val(0, x), title.split())
            for v in title_val:
                title_cnt[v] += 1
        if line_cnt % 100000 == 0:
            print "uid_cnt: %s"%len(uid_cnt)
            print "ipid_cnt: %s"%len(ipid_cnt)
            print "adid_cnt: %s"%len(adid_cnt)
            print "query_cnt: %s"%len(query_cnt)
            print "title_cnt: %s"%len(title_cnt)
            print "query_param_cnt: %s"%len(query_param_cnt)
            print "ad_param_cnt: %s"%len(ad_param_cnt)

    write_dump("data/uid_cnt.dump", uid_cnt)
    write_dump("data/ipid_cnt.dump", ipid_cnt)
    write_dump("data/adid_cnt.dump", adid_cnt)
    write_dump("data/query_cnt.dump", query_cnt)
    write_dump("data/title_cnt.dump", title_cnt)
    write_dump("data/query_param_cnt.dump", query_param_cnt)
    write_dump("data/ad_param_cnt.dump", ad_param_cnt)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--test', type=int, default=0)
    parser.add_argument('--mongo', type=int, default=0)
    parser.add_argument('--sz', type=int, default=None)
    parser.add_argument('--maxl', type=int, default=1e6)
    parser.add_argument('--type', type=str, default="ins")
    parser.add_argument('--sr', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=9)
    parser.add_argument('--date', type=int, default=0)
    args = parser.parse_args()

    if args.mongo:
        from pymongo import MongoClient
        import functools32 as functools
        client = MongoClient('localhost', 27017)
        db = client.test
        @functools.lru_cache(maxsize=1000000)
        def get_ad_info(aid):
            ad_info = db.ad_info.find_one({"AdID": aid})
            return trans_ad_info(ad_info)
    main()













