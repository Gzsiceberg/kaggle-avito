import os, argparse
import subprocess, multiprocessing

def run_cmd((cmd, log_file)):
    process = subprocess.Popen(cmd, shell=True,
                       stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT, bufsize=1)
    with open(log_file, "w", 0) as wfile:
         wfile.write("-" *80 + "\n")
         wfile.write(cmd + "\n")
         for t, line in enumerate(iter(process.stdout.readline,'')):
             line = line.rstrip()
             wfile.write(line + "\n")
    process.communicate()
    return process.returncode

def main():
    pool = multiprocessing.Pool(args.nproc)
    all_cmd = [("pypy gen_cnt_part.py --sz %s"%key, "cnt_%s.log"%key) for key in range(12)]
    pool.map_async(run_cmd, all_cmd).get(9999999)
    os.system("cd data; sort -t',' -k 1 -g -S 2G -m user_cnt_*.csv_sorted -o user_cnt.csv")
    os.system("cd data; sort -t',' -k 1 -g -S 2G -m user_aid_cnt_*.csv_sorted -o user_aid_cnt.csv")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--new', type=int, default=0)
    parser.add_argument('--nproc', type=int, default=4)
    args = parser.parse_args()
    main()