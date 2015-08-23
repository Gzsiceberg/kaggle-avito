import os, argparse
import subprocess, multiprocessing

FFM_PATH = "IceLR/ffm"
FM_PATH = "IceLR/fm"
data_cmd = {
    "ins":  "pypy ins/gen_data.py --type ins",
    "ins2": "pypy ins2/gen_data.py --type ins2",

    "ins3":    "pypy ins3/gen_data.py --type ins3 --sr 0.1 --log 0 --seed 3",
    "ins5":    "pypy ins5/gen_data.py --type ins5",
    "ins_bag": "pypy ins_bag/gen_data.py --type ins_bag",
    "ins20":   "pypy ins20/gen_data.py --type ins20 --sr 0.2",
    "xgb3":    "pypy ins3/gen_data.py --type xgb3 --sr 0.1 --log 0 --seed 3",
    "xgb5":    "pypy xgb5/gen_data.py --type xgb5"
}

train_cmd = {
    "ins":      FFM_PATH + " --passes 10 --sr 0.1 --nthread 20 --l2 1e-5 --alpha 0.25 "\
                "--train_path %s --validate_path %s --b 22 --shuffle --seed 9",

    "ins2":     FFM_PATH + " --passes 10 --sr 0.1 --nthread 20 --l2 1e-5 --alpha 0.25 "\
                "--train_path %s --validate_path %s --b 22 --shuffle --seed 9",

    "ins3":     FFM_PATH + " --passes 10 --sr 0.1 --nthread 20 --l2 1e-05 --alpha 0.25 "\
                "--train_path %s --validate_path %s --b 22 --shuffle --seed 9",

    "ins5_fm":  FM_PATH + " --passes 4 --sr 0.1 --nthread 20 --l2 1e-05 --alpha 0.1 "\
                "--train_path %s --validate_path %s --b 22 --shuffle --seed 9 --factor 80",

    "ins5":     FFM_PATH + " --passes 7 --sr 0.1 --nthread 20 --l2 3.71990308357e-06 --alpha 0.190238966372 "\
                "--train_path %s --validate_path %s --b 22 --shuffle --seed 9",

    "ins_bag":  FFM_PATH + " --passes 7 --sr 0.1 --nthread 20 --l2 5e-06 --alpha 0.25 "\
                "--train_path %s --validate_path %s --b 22 --shuffle --seed 9",

    "ins20":    FFM_PATH + " --passes 20 --sr 0.2 --nthread 20 --l2 1e-5 --alpha 0.25 "\
                "--train_path %s --validate_path %s --b 22 --shuffle --seed 9",

    "xgb3":     "python xgb.py --tr %s --te %s --model_out data/xgb3_%s.model --prob data/xgb3_%s.out --round 1000",
    "xgb5":     "python xgb.py --tr %s --te %s --model_out data/xgb5_%s.model --prob data/xgb5_%s.out --round 1000"
}

prepare_data_cmd = {
    "ad_price":   "cat data/AdsInfo.tsv | awk -F \t '{print $1,$5}' > data/ad_price.tsv",
    "split":      "pypy prepare/split_by_uid.py",
    "gen_cnt":    "pypy prepare/gen_cnt.py",
    "gen_par":    "pypy prepare/gen_params.py",
    "gen_filter": "pypy prepare/gen_filter.py",
}

ensemble_cmd = {
    "nn":   "python ensemble/stack.py --predict cv --model nn; python ensemble/stack.py --predict stack --model nn",
    "xgb":  "python ensemble/stack.py --predict cv --model xgb; python ensemble/stack.py --predict stack --model xgb",
    "mean":  "python ensemble/stack.py --predict stack --model mean",
}

def run_cmd(cmd):
    print cmd
    process = subprocess.Popen(cmd, shell=True,
                       stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT)
    for t, line in enumerate(iter(process.stdout.readline,'')):
        line = line.rstrip()
        print line
    process.communicate()
    return process.returncode

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--type', type=str)
    parser.add_argument('--method', type=str)
    parser.add_argument('--train', type=int, nargs="+", default=[])
    args = parser.parse_args()
    if not os.path.isfile("data/ad_price.tsv"):
        for k, cmd in prepare_data_cmd.items():
            run_cmd(cmd)

    if args.type == "ensemble":
        cmd = ensemble_cmd.get(args.method)
        run_cmd(cmd)

    if not os.path.isfile("data/tr.%s"%args.type) and args.type in data_cmd:
        cmd = data_cmd.get(args.type)
        run_cmd(cmd)

    if not os.path.isfile("data/all.%s"%args.type) and args.type in data_cmd:
        if args.type == "ins20":
            test_cmd = "cd data;"\
            "cat cv.%s | awk 'BEGIN{srand(9)}{if ($1==0 && rand()>0.2) next; print $0}' > cv_tr.%s;"\
            "cat tr.%s cv_tr.%s > all.%s"%tuple([args.type] * 5)
        else:
            test_cmd = "cd data;"\
            "cat cv.%s | awk 'BEGIN{srand(9)}{if ($1==0 && rand()>0.1) next; print $0}' > cv_tr.%s;"\
            "cat tr.%s cv_tr.%s > all.%s"%tuple([args.type] * 5)
        run_cmd(test_cmd)

    if 1 in args.train:
        if "xgb" in args.type:
            cmd = train_cmd.get(args.type)%("data/tr.%s"%args.type, "data/cv.%s"%args.type, "train", "train")
        else:
            if "_fm" in args.type:
                cmd = train_cmd.get(args.type)%("data/tr.ins5", "data/cv.ins5")
            else:
                cmd = train_cmd.get(args.type)%("data/tr.%s"%args.type, "data/cv.%s"%args.type)
        run_cmd(cmd)

    if 2 in args.train:
        if "xgb" in args.type:
            cmd = train_cmd.get(args.type)%("data/all.%s"%args.type, "data/te.%s"%args.type, "test", "test")
        else:
            if "_fm" in args.type:
                cmd = train_cmd.get(args.type)%("data/all.ins5", "data/te.ins5")
            else:
                cmd = train_cmd.get(args.type)%("data/all.%s"%args.type, "data/te.%s"%args.type)
        run_cmd(cmd)


















