from util import data, next_row, read_tsv, convert_ts
from csv import DictWriter
from collections import defaultdict
from datetime import datetime
import argparse, os, random

uid_sid = defaultdict(list)
uid_aid_cnt = defaultdict(lambda : [0, 0, 0])
uid_pos_cnt = defaultdict(lambda : [0, 0, 0])

def get_rows(all_se):
    l = len(all_se)
    for i in xrange(l):
        ts_i, sid_i, se_i = all_se[i]
        row = {}
        row["SearchID"] = sid_i
        row["t_cnt"] = len(all_se)
        bf_cnt, af_cnt = 0, 0
        bf_3h_cnt, af_3h_cnt = 0, 0
        bf_clk_cnt = 0

        all_clk_aid = []
        for j in xrange(l):
            if i == j:
                continue
            ts_j, sid_j, se_j = all_se[j]

            delta = ts_j - ts_i
            if ts_j - ts_i <= 0:
                bf_cnt += len(se_j)
                for impr in se_j:
                    bf_clk_cnt += impr[1]
                    if impr[1] > 0:
                        all_clk_aid.append(impr[0])
                if delta > -3600 * 3:
                    bf_3h_cnt += len(se_j)
            elif ts_j - ts_i >= 0:
                af_cnt += len(se_j)
                if delta < 3600 * 3:
                    af_3h_cnt += len(se_j)

        row["bf_cnt"] = bf_cnt
        row["bf_3h_cnt"] = bf_3h_cnt
        row["af_cnt"] = af_cnt
        row["af_3h_cnt"] = af_3h_cnt
        row["bf_clk_cnt"] = bf_clk_cnt
        sorted_clk_aid = sorted(all_clk_aid[-2:])
        row["bag2"] = abs(hash(tuple(sorted_clk_aid)))
        sorted_clk_aid = sorted(all_clk_aid[-1:])
        row["bag1"] = abs(hash(tuple(sorted_clk_aid)))
        yield row

def get_aid_rows(uid, all_se):
    all_rows = []
    for ts, sid, se in all_se:
        for impr in se:
            aid = impr[0]
            cnt_array = uid_aid_cnt[(uid, aid)]
            row = {}
            row["SearchID"] = sid
            row["AdID"] = aid
            row["clk_cnt"] = cnt_array[0]
            row["show_cnt"] = cnt_array[1]
            cnt_array[0] += impr[1]
            cnt_array[1] += 1
            cnt_array[2] += 1
            
            pos = impr[2]
            pos_cnt_array = uid_pos_cnt[(uid, pos)]
            row["pos_clk_cnt"] = pos_cnt_array[0]
            row["pos_show_cnt"] = pos_cnt_array[1]
            pos_cnt_array[0] += impr[1]
            pos_cnt_array[1] += 1
            pos_cnt_array[2] += 1

            all_rows.append(row)

    t = 0
    for ts, sid, se in all_se:
        for impr in se:
            aid = impr[0]
            cnt_array = uid_aid_cnt[(uid, aid)]
            row = all_rows[t]
            row["t_show_cnt"] = cnt_array[2]
            t += 1
            yield row

def main():
    train_iter = next_row(read_tsv("data/stream_%s.tsv"%args.sz))
    test_iter = iter([])
    sinfo_iter = read_tsv("data/sinfo_%s.tsv"%args.sz)
    del_keys_set = ["HistCTR", "SearchID", "ObjectType"]

    for t, (data_type, rows, sinfo) in enumerate(data(train_iter=train_iter, test_iter=test_iter, sinfo_iter=sinfo_iter)):
        uid = int(sinfo["UserID"])
        date_str = sinfo["SearchDate"]
        ts = convert_ts(datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S.0"))

        rows = filter(lambda x : int(x["ObjectType"]) == 3, rows)
        for row in rows:
            for key in del_keys_set:
                del row[key]
            for key in row:
                row[key] = int(row[key]) if row[key] != "" else 0
        item = (
                ts,
                int(sinfo["SearchID"]),
                tuple([(row["AdID"], row["IsClick"], row["Position"]) for row in rows]),
                )
        uid_sid[uid].append(item)

    print "uid_sid: %s"%len(uid_sid)
    for uid in uid_sid:
        uid_sid[uid].sort()

    print "start user_cnt."
    file_name = "data/user_cnt_%s.csv"%args.sz
    with open(file_name, "w") as f:
        writer = DictWriter(f, fieldnames=["SearchID", "t_cnt", "bf_cnt", "af_cnt", "bf_3h_cnt", "af_3h_cnt", "bf_clk_cnt", "bag2", "bag1"])
        writer.writeheader()
        for uid in uid_sid:
            all_se = uid_sid[uid]
            writer.writerows(get_rows(all_se))
    os.system('sort -t"," -k1 -g -S 2G %s -o %s_sorted'%(file_name, file_name))

    print "start user_aid_cnt."
    file_name = "data/user_aid_cnt_%s.csv"%args.sz
    with open(file_name, "w") as f:
        writer = DictWriter(f, fieldnames=["SearchID", "AdID", "clk_cnt", "show_cnt", "t_show_cnt", "pos_clk_cnt", "pos_show_cnt"])
        writer.writeheader()
        for uid in uid_sid:
            all_se = uid_sid[uid]
            writer.writerows(get_aid_rows(uid, all_se))
    os.system('sort -t"," -k1 -g -S 2G %s -o %s_sorted'%(file_name, file_name))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--sz', type=int, default=0)
    args = parser.parse_args()

    main()