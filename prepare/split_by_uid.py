from util import data
from collections import defaultdict
from csv import DictWriter

def main():
    sz = 12
    dt_fh_list = [None for i in range(sz)]
    si_fh_list = [None for i in range(sz)]
    close_fn = []
    for t, (data_type, rows, sinfo) in enumerate(data()):
        uid = int(sinfo["UserID"])
        m = uid % sz
        for row in rows:
            if "IsClick" not in row:
                row["IsClick"] = 0
            if "ID" in row:
                del row["ID"]
        if dt_fh_list[m] is None:
            fh = open("data/stream_%s.tsv"%m, "w")
            close_fn.append(fh)
            dt_fh_list[m] = DictWriter(fh, delimiter='\t', fieldnames=rows[0].keys())
            dt_fh_list[m].writeheader() 
        if si_fh_list[m] is None:
            fh = open("data/sinfo_%s.tsv"%m, "w")
            close_fn.append(fh)
            si_fh_list[m] = DictWriter(fh, delimiter='\t', fieldnames=sinfo.keys()) 
            si_fh_list[m].writeheader()
        dt_fh, si_fh = dt_fh_list[m], si_fh_list[m]
        si_fh.writerow(sinfo)
        dt_fh.writerows(rows)

    for fh in close_fn:
        fh.close()

if __name__ == '__main__':
    main()