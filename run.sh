pypy run.py --type ins --train 1 2 > ins_run.log
pypy run.py --type ins2 --train 1 2 > ins2_run.log
pypy run.py --type ins5 --train 1 2 > ins5_run.log
pypy run.py --type ins5_fm --train 1 2 > ins5_fm_run.log
pypy run.py --type ins3 --train 1 2 > ins3_run.log
pypy run.py --type ins_bag --train 1 2 > ins_bag_run.log
pypy run.py --type ins20 --train 1 2 > ins20_run.log
pypy run.py --type xgb5 --train 1 2 > xgb5_run.log
pypy run.py --type xgb3 --train 1 2 > xgb3_run.log
cat data/cv.ins5 | awk '{print $1}' > data/cv_label.out

# pypy run.py --type ensemble --method nn > nn_log.log
# pypy run.py --type ensemble --method xgb > xgb_log.log
# pypy run.py --type ensemble --method mean > mean_log.log
# pypy ensemble/merge.py



