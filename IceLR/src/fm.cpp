#include <iostream>
#include <string>
#include <vector>
#include <gflags/gflags.h>
#include <cmath>
#include "common.hpp"
#include "timer.h"
#include <tbb/concurrent_hash_map.h>
#include <omp.h>
#include <tbb/atomic.h>
#include <boost/functional/hash.hpp>
#include <exception>
#include "model.pb.h"
#include <fstream>
#include <memory> 
#include "fm.hpp"

DEFINE_string(train_path, "train.ice", "train file path.");
DEFINE_string(validate_path, "validate.ice", "validate file path.");
DEFINE_int32(b, 20, "hash size");
DEFINE_int32(seed, -1, "random seed");
DEFINE_int32(factor, 4, "model factor");
DEFINE_string(alpha, "1e-1", "alpha");
DEFINE_string(l2, "1e-6", "lambda 2");
DEFINE_double(sr, 1.0, "sample rate");
DEFINE_bool(test, false, "only test");
DEFINE_bool(predict, true, "need predict");
DEFINE_string(model_in, "", "model load path");
DEFINE_string(model_out, "", "model dump path");

// need be used in other files
DEFINE_string(slot_mask, "", "slot mask");
DEFINE_int32(passes, 1, "passes");
DEFINE_int32(nthread, 2, "thread number");
DEFINE_bool(shuffle, false, "shuffle the data");
DEFINE_bool(debug, false, "debug");

int main(int argc, char *argv[])
{
    google::ParseCommandLineFlags(&argc, &argv, true);
    if(FLAGS_seed > 0)
        std::srand(FLAGS_seed);
    using Estimate = FFMEstimate<BasicParameter>;
    std::unique_ptr<Estimate> learner(new Estimate());
    try
    {
        if(!FLAGS_model_in.empty())
        {
            learner->load(FLAGS_model_in);
        }
        Problem tr_prob;
        Problem va_prob(FLAGS_validate_path);
        std::vector<Problem> prob_vec;
        if(!FLAGS_test)
        {
            tr_prob.load_data(FLAGS_train_path);
            learner->fit(tr_prob, va_prob);
            if(!FLAGS_model_out.empty())
                learner->dump(FLAGS_model_out);
        }

        if(FLAGS_predict)
        {
            double sum_prob = 0.0;
            uint32_t validate_size = va_prob.all_y.size();
            FILE *file = nullptr;
            if(validate_size > 0)
            {
                std::string pred_path = FLAGS_validate_path + ".ffm.out";
                printf("start otuput predict to %s\n", pred_path.c_str());
                fflush(stdout);
                file = open_c_file(pred_path, "w");
                for(auto i = 0; i < validate_size; ++i)
                {
                    auto &f = va_prob.all_f[i];
                    double prob = learner->predict(f, 0, false, 2);
                    sum_prob += prob;
                    fprintf(file, "%lf\n", prob);
                }
                printf("average prob: %lf\n", sum_prob/validate_size);
                learner->print_stats();
                fflush(stdout);
            }
            else
            {
                std::string pred_path = FLAGS_train_path + ".ffm.out";
                printf("start otuput predict to %s\n", pred_path.c_str());
                fflush(stdout);
                file = open_c_file(pred_path, "w");

                uint32_t train_size = tr_prob.all_y.size();
                if(train_size == 0)
                    tr_prob.load_data(FLAGS_train_path);
                for(auto i = 0; i < train_size; ++i)
                {
                    auto &f = tr_prob.all_f[i];
                    double prob = learner->predict(f);
                    sum_prob += prob;
                    fprintf(file, "%lf\n", prob);
                }
                printf("average prob: %lf\n", sum_prob/train_size);
                fflush(stdout);
            }
            fclose(file);
        }
        return 0;
    }
    catch (std::exception const &exc)
    {
        std::cerr << "Exception caught " << exc.what() << "\n";
    }
}
