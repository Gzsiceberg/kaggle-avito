#ifndef _FFM_H
#define _FFM_H

#include <iostream>
#include <string>
#include <vector>
#include <gflags/gflags.h>
#include <unordered_map>
#include <cmath>
#include "common.hpp"
#include "timer.h"
#include <omp.h>
#include <boost/functional/hash.hpp>
#include <exception>
#include "model.pb.h"
#include <fstream>
#include <memory> 
#include "hash_filter.hpp"
#include "auc.hpp"
#include "random.hpp"
#include <pmmintrin.h>

DECLARE_int32(passes);
DECLARE_int32(nthread);
DECLARE_int32(factor);
DECLARE_int32(b);

DECLARE_string(slot_mask);
DECLARE_string(l2);
DECLARE_string(alpha);

DECLARE_double(sr);
DECLARE_bool(shuffle);
DECLARE_bool(debug);
DECLARE_bool(early_stop);

using std::vector;
using std::string;

void print_m128(__m128 d, string prefix)
{
    float res[4];
    _mm_store_ps(res, d);
    printf("%s: %f %f %f %f\n", prefix.c_str(), res[0], res[1], res[2], res[3]);
}

class MetaParameter
{
public:
    MetaParameter()
    {
        D = FLAGS_b;
        n_factor = FLAGS_factor;
        printf("factor: %d\n", n_factor);
        printf("lambda2: %s\nalpha: %s\nD: %d\n", FLAGS_l2.c_str(), FLAGS_alpha.c_str(), D);
        vector<uint32_t> slot_mask;
        parse_line(FLAGS_slot_mask.begin(), FLAGS_slot_mask.end(), slot_mask);

        printf("slot_mask: ");
        slot.resize(1024, false);
        for(auto it : slot_mask)
        {
            printf("%d ", it);
            slot[it] = true;
        }
        printf("\n");

        parse_line(FLAGS_l2.begin(), FLAGS_l2.end(), l2);
        parse_line(FLAGS_alpha.begin(), FLAGS_alpha.end(), alpha);

    }

    double beta = 1.0;
    vector<bool> slot;
    vector<double> l2, alpha;
    int32_t D;
    int32_t n_factor = 0;
};

class BasicParameter
{
public:
    BasicParameter(){};
    ~BasicParameter(){};

    void init(MetaParameter &meta_parameter)
    {
        if(!wv.empty()) return;
        float const coef = static_cast<float>(1.0/sqrt(static_cast<double>(meta_parameter.n_factor)));
        wv.resize(meta_parameter.n_factor, 0);
        sgv.resize(meta_parameter.n_factor, 0);
        for(uint32_t d = 0; d < wv.size(); ++d)
        {
            wv[d] = coef * static_cast<float>(drand48() - 0.5);
            sgv[d] = meta_parameter.beta;
        }
    }

    const float* get_value(MetaParameter &meta_parameter) const
    {
        return wv.data();
    }

    void update_value(float* delta, MetaParameter &meta_parameter, uint32_t slot)
    {
        float* w = wv.data(); 
        float* sg = sgv.data(); 
        auto& alpha = meta_parameter.alpha;
        auto& l2 = meta_parameter.l2;
        __m128 const XMMeta = _mm_set1_ps(alpha.size() == 1? alpha[0] : alpha[slot]);
        __m128 const XMMlambda = _mm_set1_ps(l2.size() == 1? l2[0] : l2[slot]);
        for(int i = 0; i < wv.size(); i += 4)
        {
            __m128 XMMw1 = _mm_load_ps(w + i);
            __m128 XMMwg1 = _mm_load_ps(sg + i);
            __m128 XMMg1 = _mm_add_ps(_mm_load_ps(delta + i), 
                _mm_mul_ps(XMMlambda, XMMw1));

            XMMwg1 = _mm_add_ps(XMMwg1, _mm_mul_ps(XMMg1, XMMg1));
            XMMw1 = _mm_sub_ps(XMMw1, _mm_mul_ps(XMMeta, 
                    _mm_mul_ps(_mm_rsqrt_ps(XMMwg1), XMMg1)));

            _mm_store_ps(w + i, XMMw1);
            _mm_store_ps(sg + i, XMMwg1);
        }
    }

    vector<float> wv;
    vector<float> sgv;
};


template <class PT>
class FFMEstimate : public BaseEstimate
{
public:
    template<class T> using ParameterMap = std::unordered_map<uint32_t, T>;

    FFMEstimate()
    {
        model_map.resize(1);
    }
    ~FFMEstimate(){};

    void init_model(Problem &tr_prob)
    {
        printf("initialize...\n");
        auto train_size = tr_prob.all_y.size();
        for(unsigned i = 0; i < train_size; ++i)
        {
            double y = tr_prob.all_y[i];
            auto &f = tr_prob.all_f[i];
            int index = 0;
            for(unsigned j = 0; j < f.size(); ++j) 
            {
                model_map[index][f[j]].init(meta_parameter);
            }
        }
    }

    double evaluate(uint32_t iter, std::vector<double> &tr_p, std::vector<double> &va_p, 
        Problem &tr_prob, Problem &va_prob, Timer &timer)
    {
        double tr_loss = 0.0, va_loss = 0.0;
        // start to evaluate metrics.
        auto train_size = tr_prob.all_y.size();
        for (int i = 0; i < train_size; ++i)
        {
            double y=tr_prob.all_y[i], p=tr_p[i];
            tr_loss += -log(std::max(p, 1e-12));
        }
        tr_loss /= train_size;

        auto validate_size = va_prob.all_y.size();
        for(int i = 0;i < validate_size; ++i)
        {
            double y=va_prob.all_y[i], p = va_p[i];
            va_loss += -log(std::max(p, 1e-12));
        }
        va_loss /= validate_size;
        printf("%4d%9.1f%15.6f%15.6f\n", iter, timer.toc(), 
            tr_loss, va_loss);
        fflush(stdout);
        return va_loss;
    }

    void fit(Problem &tr_prob, Problem &va_prob)
    {
        init_model(tr_prob);
        printf("%4s%9s%15s%15s\n", 
            "iter", "time", 
            "tr_loss", "va_loss");
        fflush(stdout);
        
        size_t train_size = tr_prob.all_y.size();
        size_t validate_size = va_prob.all_y.size();
        std::vector<double> tr_p, va_p;
        tr_p.resize(train_size, 0);
        va_p.resize(validate_size, 0);

        std::vector<uint32_t> order(train_size);
        for(uint32_t i = 0; i < train_size; ++i)
        {
            order[i] = i;
        }
        Timer timer;
        omp_set_num_threads(FLAGS_nthread);
        bool early_stop = false;
        for(unsigned iter = 0; iter < FLAGS_passes && !early_stop; ++iter)
        {
            if(FLAGS_shuffle)
            {
                std::random_shuffle(order.begin(), order.end());
            }
            #pragma omp parallel for schedule(static)
            for(unsigned j = 0; j < train_size; ++j)
            {
                auto i = order[j];
                double y = tr_prob.all_y[i];
                auto &f = tr_prob.all_f[i];
                int index = 0;
                double p = predict(f, index, true);
                tr_p[i] = y < 1 ? 1 - p : p;
                apply_all(f, false, index, p - y);
            }

            #pragma omp parallel for schedule(static)
            for(unsigned i = 0; i < validate_size; ++i) 
            {
                auto &f = va_prob.all_f[i];
                double y = va_prob.all_y[i];
                int index = 0;
                double p = predict(f);
                p = y < 1 ? 1 - p : p;
                va_p[i] = p;
            }

            double va_loss = evaluate(iter, tr_p, va_p, tr_prob, va_prob, timer);
        }
        omp_set_num_threads(1);
    }

    double predict(std::vector<uint32_t> &f, int index = 0, bool train=false, int32_t predict=1)
    {
        double wtx = apply_all(f, predict, index, 0.0);
        return 1.0f/(1.0f + exp((train ? log(FLAGS_sr): 0) - wtx));
    }

    const float* pull(uint32_t x, int index = 0)
    {
        static vector<float> empty_w(1024, 0.0);
        auto itr = model_map[index].find(x);
        if(itr != model_map[index].end())
        {
            return itr->second.get_value(meta_parameter);
        }
        return empty_w.data();
    }

    void push(uint32_t x, int index, float* delta)
    {
        model_map[index][x].update_value(delta, meta_parameter, 0);
    }

    double apply_all(std::vector<uint32_t> &f, int32_t predict=1, int index=0, double delta=0)
    {
        int32_t n_factor = meta_parameter.n_factor;
        float v = 1.0/f.size();
        __m128 const XMMv = _mm_set1_ps(v);
        __m128 const XMMkappav = _mm_set1_ps(delta * v);

        if(all_w_ptr == nullptr)
        {
            all_w_ptr = new vector<const float*>(1024);
        }
        auto& all_w = *all_w_ptr;
        for(unsigned j = 0; j < f.size(); ++j)
        {
            all_w[j] = pull(f[j], index);
        } 

        if(all_grads_ptr == nullptr)
        {
            all_grads_ptr = new vector<vector<float>>(1024);
            for(int i = 0; i < all_grads_ptr->size(); ++i)
            {
                (*all_grads_ptr)[i].resize(n_factor, 0.0);
            }
        }

        auto& all_grads = *all_grads_ptr;
        float* sg = all_grads[1023].data();
        for(unsigned k = 0; k < n_factor; ++k) sg[k] = 0;
        for(unsigned k = 0; k < n_factor; k += 4)
        {
            __m128 XMMsw = _mm_load_ps(sg + k);
            for(unsigned j = 0; j < f.size(); ++j) 
            {
                XMMsw = _mm_add_ps(XMMsw, _mm_load_ps(all_w[j] + k));
            }
            _mm_store_ps(sg + k, XMMsw);
        }

        __m128 XMMwtx = _mm_setzero_ps();
        for(unsigned j = 0; j < f.size(); ++j)
        {
            const float* x_w = all_w[j]; 
            float* g = all_grads[j].data();
            for(unsigned k = 0; k < n_factor; k += 4)
            {
                const __m128 XMMw = _mm_load_ps(x_w + k);
                const __m128 XMMsw = _mm_load_ps(sg + k);
                if(j == 0)
                {
                    XMMwtx = _mm_add_ps(XMMwtx, _mm_mul_ps(XMMsw, XMMsw));
                }
                XMMwtx = _mm_sub_ps(XMMwtx, _mm_mul_ps(XMMw, XMMw));
                if(predict == 0)
                {
                    _mm_store_ps(&all_grads[j][k], _mm_mul_ps(_mm_sub_ps(XMMsw, XMMw), XMMkappav));
                }
            }
        }
        XMMwtx = _mm_hadd_ps(XMMwtx, XMMwtx);
        XMMwtx = _mm_hadd_ps(XMMwtx, XMMwtx);
        float t;
        _mm_store_ss(&t, XMMwtx);
        t *= 0.5 * v;

        if(predict == 0)
        {
            for(int i = 0; i < f.size(); ++i)
            {
                push(f[i], index, all_grads[i].data());
            }
        }
        return t;
    }

    void print_stats()
    {
    }

    void dump(std::string& path)
    {
        printf("dump model in %s\n", path.c_str());
        fflush(stdout);
    }

    void load(std::string& path)
    {
        printf("load model from %s\n", path.c_str());
        fflush(stdout);
    }

    /* data */
    MetaParameter meta_parameter;
    vector<ParameterMap<PT>> model_map;
    static __thread vector<const float*>* all_w_ptr;
    static __thread vector<vector<float>>* all_grads_ptr;
};

template <typename PT>
__thread vector<const float*>* FFMEstimate<PT>::all_w_ptr=nullptr;

template <typename PT>
__thread vector<vector<float>>* FFMEstimate<PT>::all_grads_ptr=nullptr;

#endif