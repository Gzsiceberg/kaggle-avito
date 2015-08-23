from util import read_tsv
from csv import DictWriter
import re
import ast, json

type_set = set()
def main():
    search_par_h = open("data/search_params.csv", "w")
    writer = DictWriter(search_par_h, fieldnames=["SearchID", "SearchParams"])
    writer.writeheader()
    for t, row in read_tsv("data/SearchInfo.tsv"):
        sparams = row["SearchParams"]
        if not sparams:
            continue
        sid = int(row["SearchID"])
        sparams = re.sub(r"([A-Za-z0-9]+):", r'"\1":', sparams)
        sparams = sparams.replace("'", "\"")
        sparams = sparams.replace("Минивэн\",", "\"Минивэн\",")
        sparams = sparams.replace("Микроавтобус\"]", "\"Микроавтобус\"]")
        sparams = unicode(sparams, "utf-8")
        try:
            sparams = json.loads(sparams)
            for k, v in sparams.items():
                t = type(v)
                if t not in type_set:
                    print t, k, v
                    type_set.add(t)
            sparams_str = json.dumps(sparams)
            writer.writerow({"SearchID": sid, "SearchParams": sparams_str})
        except Exception as e:
            print e
            print sparams

if __name__ == '__main__':
    main()