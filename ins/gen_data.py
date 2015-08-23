from csv import DictReader
from datetime import datetime, timedelta
from collections import defaultdict
import cPickle as pickle
from math import exp, log, sqrt
import random, gc
from util import read_dump, write_dump, cache, read_tsv, convert_ts, data, next_row, get_category
import argparse, ast, re, json

def filter_row(row, data_type):
    object_type = int(row["ObjectType"])
    if object_type != 3:
        return False
    y = int(row.get("IsClick", 0))
    if data_type == 0 and y == 0 and random.random() > args.sr:
        return False
    return True

def calc_ctr(x, y):
    avg_ctr = 0.0060281
    return int(round((x + avg_ctr * 10) * 100.0 / (y + 10)))

def log_trans(x):
    return int(round(log(x + 1)))

def get_user_info():
    user_info_map = {}
    for t, row in read_tsv("data/UserInfo.tsv"):
        for k in row:
            row[k] = int(row[k])
        uid = row["UserID"]
        del row["UserID"]
        user_info_map[uid] = row
    return user_info_map

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

ad_price_list = []
ad_price_iter = read_tsv("data/ad_price.tsv", delimiter=" ")
def get_ad_price(aid):
    while aid - 1 >= len(ad_price_list):
        t, row = next(ad_price_iter, (None, None))
        if row is None:
            break
        price = row["Price"]
        price = float(price) if price else ""
        ad_price_list.append(price)
    return ad_price_list[aid - 1]

def get_features(sinfo, rows, test=False):
    feature_map = defaultdict(list)
    sid = sinfo["SearchID"]
    sinfo["SearchParams"] = get_se_param(sid)

    user_cnt_row = next(user_cnt_iter, (None, None))[1]
    while int(user_cnt_row["SearchID"]) != sid:
        user_cnt_row = next(user_cnt_iter, (None, None))[1]

    user_aid_cnt_rows = next(user_aid_cnt_iter, (None, None))[1]
    while int(user_aid_cnt_rows[0]["SearchID"]) != sid:
        user_aid_cnt_rows = next(user_aid_cnt_iter, (None, None))[1]

    user_aid_cnt_dict = {}
    for row in user_aid_cnt_rows:
        aid = int(row["AdID"])
        user_aid_cnt_dict[aid] = row

    ad_infos = []
    for row in rows:
        aid = int(row["AdID"])
        row.update(user_aid_cnt_dict[aid])
        ad_infos.append(get_ad_info(aid))

    uid = int(sinfo["UserID"])
    user_info = user_info_map.get(uid, {"UserAgentID": "",
                                        "UserAgentOSID": "",
                                        "UserDeviceID": "",
                                        "UserAgentFamilyID": ""})
    feature_map["user_cnt"] = [user_cnt_row]
    feature_map["user_info"] = [user_info]
    feature_map["ad_info"] = ad_infos
    feature_map["stream_info"] = rows
    feature_map["sinfo"] = [sinfo]
    return feature_map

def extract_slot_feas(rows, sinfo):
    data = map(lambda x: (int(x["Position"]), int(x["ObjectType"]), x), rows)
    data.sort()

    price_data = []
    ot_cnt = defaultdict(int)
    all_pos = []
    all_ot = []
    for i in range(len(data)):
        all_pos.append(data[i][0])
        all_ot.append(data[i][1])
        
        aid = int(data[i][2]["AdID"])
        price_data.append((get_ad_price(aid), i))
        i_obt = data[i][1]
        ot_cnt[i_obt] += 1

        ucnt, lcnt = 0, 0
        for j in range(len(data)):
            if i == j:
                continue
            j_obt = data[j][1]
            if j_obt == 2:
                if i < j:
                    lcnt += 1
                else:
                    ucnt += 1
        data[i][2]["hl_lcnt"] = lcnt
        data[i][2]["hl_ucnt"] = ucnt

    for k in range(1, 4):
        v = ot_cnt[k]
        sinfo["ot%s_cnt"%k] = v
    sinfo["record_cnt"] = len(rows)
    sinfo["pos_type"] = hash_val(0, tuple(all_pos))
    sinfo["pos_ot_type"] = hash_val(0, tuple(all_ot))

    price_data.sort()
    avg_price, avg_cnt = 0, 0
    for p, i in price_data:
        if p != "":
            avg_price += p
            avg_cnt += 1
            data[i][2]["price_pos"] = i
        else:
            data[i][2]["price_pos"] = -1
    if avg_cnt == 0 or avg_price <= 0:
        pass
    else:
        avg_price /= avg_cnt
    for p, i in price_data:
        if not p:
            ratio = -1
        elif avg_price <= 0:
            ratio = -2
        else:
            ratio = int(round((p / avg_price) * 100))
        data[i][2]["price_ratio"] = ratio

def stream_info_func(vs, name=False):
    keys = ["AdID", 
            "Position", 
            "HistCTR", 
            "hl_lcnt", 
            "hl_ucnt",

            "clk_cnt",
            "show_cnt",
            "t_show_cnt",

            "price_pos",
            "price_ratio",
            ]
    for v in vs[0]:
        if name:
            yield keys
        else:
            x = {}
            for k in keys:
                if k == "HistCTR":
                    val = v[k]
                    if val != "":
                        val = int(round(float(val) * 1000))
                elif k in ("pos_show_cnt",):
                    val = log_trans(int(v[k]))
                else:
                    val = v[k]
                x[k] = val
            x["u_aid_ctr"] = calc_ctr(int(x["clk_cnt"]), int(x["show_cnt"]))
            # x["u_pos_ctr"] = calc_ctr(int(x["pos_clk_cnt"]), int(x["pos_show_cnt"]))
            yield x

def sinfo_func(vs, name=False):
    keys = [
            "IPID", 
            "UserID", 
            "IsUserLoggedOn",
            "SearchQuery",
            # "SearchParams",

            "ot1_cnt",
            "ot2_cnt",
            "ot3_cnt",
            "record_cnt",

            "pos_type",
            "pos_ot_type",

            "s_LocationID", 
            "s_CategoryID",
            ]
    for v in vs[0]:
        if name:
            yield keys
        else:
            x = {}
            for k in keys:
                if k == "SearchQuery":
                    query = unicode(v["SearchQuery"], "utf-8")
                    val = map(lambda x : hash_val(0, x), query.split())
                    if len(val) == 0:
                        val = [-1,]
                else:
                    val = v[k]
                x[k] = val
            # date_str = v["SearchDate"]
            # d = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S.0")
            # x["hour"] = d.hour
            # x["weekday"] = d.weekday()
            yield x

def user_info_func(vs, name=False):
    for v in vs[0]:
        if name:
            yield v.keys()
        else:
            x = {}
            for k in v:
                val = v[k]
                x[k] = val
            yield x

def ad_info_func(vs, name):
    keys = ["CategoryID", 
            "Price", 
            "Params",
            ]
    for v in vs[0]:
        if name:
            yield keys
        else:
            x = {}
            for k in keys:
                val = v[k]
                if k == "Price" and val != "":
                    val = int(round(log(val + 1)))
                x[k] = val
            yield x

def ngram(query_word):
    sz = len(query_word)
    res = []
    for i in range(sz - 1):
        res.append(u"%s %s"%(query_word[i], query_word[i + 1]))
    return res

def calc_sim(qw, tw):
    cnt = 0
    min_pos = 10000
    match_w = []
    for w in qw:
        if w in tw:
            match_w.append(w)
            cnt += 1
            min_pos = min(min_pos, tw.index(w))
    ratio = int(round((cnt* 1.0 / len(qw)) * 100)) 
    return {"cnt": cnt,
            "pos": min_pos,
            "ratio": ratio,
            "match_w": match_w}

def query_feas(query_word, ad_info, name=False):
    if name:
        return ["qe_w_cnt", "qe_w_ratio", "qe_w_pos",
                "qe_ng_cnt", "qe_ng_ratio", "qe_ng_min_pos", "t_match"]
    x = {}
    title = ad_info["Title"].split()
    x["title_len"] = len(title)
    if " ".join(query_word) in ad_info["Title"]:
        x["t_match"] = 1
    else:
        x["t_match"] = 0
    # title = ad_info["Title"]
    # title_val = map(lambda x : hash_val(0, x), title.split())
    # x["title"] = title_val if title_val else [-1,]

    if len(query_word) == 0:
        x["qe_w_cnt"] = -1
        x["qe_w_ratio"] = -1
        x["qe_w_pos"] = -1
    else:
        sim = calc_sim(query_word, title)
        x["qe_w_cnt"] = sim["cnt"]
        x["qe_w_ratio"] = sim["ratio"]
        x["qe_w_pos"] = sim["pos"]
        x["match_w"] = sim["match_w"]

    qw_ngram = ngram(query_word)
    if len(qw_ngram) == 0:
        x["qe_ng_cnt"] = -1
        x["qe_ng_ratio"] = -1
        x["qe_ng_min_pos"] = -1
    else:
        title_ngram = ngram(title)
        sim = calc_sim(qw_ngram, title_ngram)
        x["qe_ng_cnt"] = sim["cnt"]
        x["qe_ng_ratio"] = sim["ratio"]
        x["qe_ng_min_pos"] = sim["pos"]
        x["match_ng"] = sim["match_w"]
    return x 

unmatch_set = set()
def param_feas(se_params, ad_info, name=False):
    if name:
        return ["par_match", "par_nmatch", "par_miss"]
    if len(se_params) == 0:
        return [-1, -1, -1]
    ad_params = ad_info["Params"]
    x = {}
    par_match = 0
    par_miss = 0
    par_nmatch = 0
    for par_key, par_v in se_params.items():
        ad_v = ad_params.get(par_key)
        if par_v == ad_v:
            par_match += 1
        elif ad_v is not None:
            par_nmatch += 1
            key =  (type(par_v), type(ad_v))
            if key not in unmatch_set:
                unmatch_set.add(key)
                print key
                print par_v.encode("utf-8"), u"----", ad_v.encode("utf-8")
        else:
            par_miss += 1
    x["par_match"] = par_match
    x["par_nmatch"] = par_nmatch
    x["par_miss"] = par_miss
    return x

def match_info_func(vs, name):
    ad_infos = vs[0]
    sinfo = vs[1][0]

    s_ca_id = int(sinfo.get("CategoryID"))
    s_ca_pid = category_map[s_ca_id]
    # se_params = sinfo["Params"]
    query = unicode(sinfo["SearchQuery"], "utf-8")
    query = query.split() if query else []

    keys = [
            "ca_match", 
            "ca_pid_match"
            ] + query_feas(None, None, True)
            # + param_feas(None, None, True)
    for ad_info in ad_infos:
        if name:
            yield keys
        else:
            x = {}

            """ ca_match """
            ca_id = int(ad_info.get("CategoryID", -1))
            if ca_id == s_ca_id:
                x["ca_match"] = ca_id
            else:
                x["ca_match"] = -1

            ca_pid = category_map.get(ca_id, -1)
            if ca_pid == s_ca_pid:
                x["ca_pid_match"] = ca_pid
            else:
                x["ca_pid_match"] = -1

            x.update(query_feas(query, ad_info))
            # x.update(param_feas(se_params, ad_info))
            yield x

log_cnt_keys = set(["t_cnt","bf_cnt","af_cnt"])
def user_cnt_func(vs, name):
    keys = ["t_cnt","bf_cnt","af_cnt",
            "bf_3h_cnt","af_3h_cnt", 
            "bf_clk_cnt",]
    for v in vs[0]:
        if name:
            yield keys
        else:
            x = {}
            for k in keys:
                if k not in v:
                    continue
                val = v[k]
                if k in log_cnt_keys:
                    val = log_trans(int(val))
                x[k] = val
            x["bf_ctr"] = calc_ctr(int(v["bf_clk_cnt"]), int(v["bf_cnt"]))
            yield x

log_cnt_keys = set(["t_cnt","bf_cnt","af_cnt"])
def new_user_cnt_func(vs, name):
    keys = ["t_cnt","bf_cnt","af_cnt",
            "bf_3h_cnt","af_3h_cnt", 
            "bf_clk_cnt", "bag"]
    for v in vs[0]:
        if name:
            yield keys
        else:
            x = {}
            for k in keys:
                if k not in v:
                    continue
                val = v[k]
                if k in log_cnt_keys:
                    val = log_trans(int(val))
                x["new_" + k] = val
            x["new_bf_ctr"] = calc_ctr(int(v["bf_clk_cnt"]), int(v["bf_cnt"]))
            yield x    

extract_func = [
    (stream_info_func, ["stream_info"]),
    (sinfo_func, ["sinfo"]),
    (user_info_func, ["user_info"]),
    (ad_info_func, ["ad_info"]),
    # (new_user_cnt_func, ["new_user_cnt"]),
    (user_cnt_func, ["user_cnt"]),
    (match_info_func, ["ad_info", "sinfo"]),
    ]

def extract(feature_map, name=False):
    ins_size = 0
    for k, v in feature_map.items():
        ins_size = max(ins_size, len(v))
    instances = [{} for _ in xrange(ins_size)]
    for func, in_keys in extract_func:
        vls = map(lambda k:feature_map[k], in_keys)
        msize = reduce(lambda x, y: max(x, len(y)), vls, 0)
        if msize == 1:
            for x in func(vls, name):
                for ins in instances:
                    ins.update(x)
                break
        else:
            for t, x in enumerate(func(vls, name)):
                instances[t].update(x)
    return instances

def hash_val(t, v, dtype=None, D=22):
    if dtype == "xgb":
        return u"%s:%s"%(t, v)
    else:
        return (t << D) | (hash(unicode(v)) & ((1 << D) - 1))

def main():
    random.seed(args.seed)
    xgb_set =set([
        "pos_type", "price_pos", "ot1_cnt", "pos_ot_type",
        "bf_cnt", "bf_clk_cnt", "u_aid_ctr", "record_cnt",
        "show_cnt", "clk_cnt", "t_cnt", "qe_w_pos",
        "HistCTR", "qe_ng_min_pos", "t_show_cnt", "bf_ctr",
        "ot2_cnt", "Price", "qe_ng_cnt", "title_len",
        "hl_ucnt", "price_ratio", "hl_lcnt", "t_match",
        "qe_w_ratio", "qe_ng_ratio", "ca_match", "Position",
        "bf_3h_cnt", "qe_w_cnt", "af_cnt", "ot3_cnt",
        "ca_pid_match", "af_3h_cnt",
        ])
    if args.test:
        fh_list = [ open("data/tr_%s.%s"%(args.test, args.type), "w"), 
                    open("data/cv_%s.%s"%(args.test, args.type), "w"), 
                    open("data/te_%s.%s"%(args.test, args.type), "w")]
    else:
        fh_list = [open("data/tr.%s"%(args.type), "w"), 
                    open("data/cv.%s"%(args.type), "w"), 
                    open("data/te.%s"%(args.type), "w")]

    data_iter = data(args.test, maxlines=args.maxl)
 
    avg_ctr = defaultdict(lambda : [0, 0])
    for line_cnt, (data_type, rows, sinfo) in enumerate(data_iter):
        sinfo["s_LocationID"] = int(sinfo["LocationID"])
        sinfo["s_CategoryID"] = int(sinfo["CategoryID"])
        extract_slot_feas(rows, sinfo)
        rows = filter(lambda x: filter_row(x, data_type), rows)
        if not rows:
            continue
        feature_map = get_features(sinfo, rows, data_type > 0)
        instances = extract(feature_map)
        if line_cnt == 0:
            for k, feas in feature_map.items():
                print "-" * 80
                print k
                print feas[0].keys()
            feas_name = sorted(instances[0].keys())
            print len(feas_name), feas_name
            if args.sz is not None:
                write_dump("feas_name.dump", feas_name)
            elif args.test:
                write_dump("feas_name%s.dump"%args.test, feas_name)
            else:
                write_dump("feas_name.dump", feas_name)

        fh = fh_list[data_type]
        for ins_map, row in zip(instances, rows):
            y = int(row.get("IsClick", 0))
            avg_ctr[data_type][0] += y
            avg_ctr[data_type][1] += 1
            ins = []
            for kt, k in enumerate(feas_name):
                if args.type == "xgb" and k not in xgb_set:
                    continue
                feas = ins_map[k]
                if line_cnt == 0:
                    print kt, k, type(feas), feas
                if isinstance(feas, list) or isinstance(feas, tuple):
                    for f in feas:
                        ins.append(hash_val(kt + 1, f, args.type))
                else:
                    ins.append(hash_val(kt + 1, feas, args.type))
            fh.write(unicode(y) + " " + " ".join(map(unicode, ins)) + "\n")
    for key, value in avg_ctr.items():
        print "%s, %s"%(key, value[0] * 1. / value[1])
    for fh in fh_list:
        fh.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--test', type=int, default=0)
    parser.add_argument('--mongo', type=int, default=0)
    parser.add_argument('--sz', type=int, default=None)
    parser.add_argument('--maxl', type=int, default=1e6)
    parser.add_argument('--type', type=str, default="ins")
    parser.add_argument('--sr', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=9)
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

    user_info_map = get_user_info()
    category_map = get_category()
    user_cnt_iter = read_tsv("data/user_cnt.csv", delimiter=",")
    user_aid_cnt_iter = next_row(read_tsv("data/user_aid_cnt.csv", delimiter=","))
    main()













