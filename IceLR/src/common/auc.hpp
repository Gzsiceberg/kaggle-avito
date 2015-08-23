#ifndef _AUC_H
#define _AUC_H

#include <cmath>
#include <string.h>
#include <stdio.h>
#include <vector>

class AUC
{
public:
    AUC(int size): _bin_size(size){
        ptable.resize(size, 0);
        ntable.resize(size, 0);
    }

    ~AUC(){}

    void clear()
    {
        for(auto i = 0; i < ptable.size(); ++i)
            ptable[i] = 0;
        for(auto i = 0; i < ntable.size(); ++i)
            ntable[i] = 0;
    }

    inline void add_data(int label, double pred)
    {
        if (pred < 0 || pred > 1) return;
        int key = pred * (_bin_size - 1);
        if (label == 1) ptable[key] ++;
        else ntable[key] ++;
    }

    void merge(const AUC& other_auc)
    {
        const double multiplier = (double)_bin_size / other_auc._bin_size;
        for (int i = 0; i < other_auc._bin_size; i ++) {
            ptable[int(i * multiplier)] += other_auc.ptable[i];
            ntable[int(i * multiplier)] += other_auc.ntable[i];
        }
    }

    void calc_auc(double& auc)
    {
        double area, fp, tp;
        area = fp = tp = 0;
        for (int i = _bin_size - 1; i >= 0; i--) {
            double newfp = fp + ntable[i];
            double newtp = tp + ptable[i];

            area += (newfp - fp) * (tp + newtp) / 2;
            fp = newfp;
            tp = newtp;
        }

        if (fp == 0 || tp == 0) {
            auc = 0;
        } else {
            auc = area / (fp * tp);
        }
    }

private:
    const int _bin_size;
    std::vector<int> ptable, ntable;
};

#endif
