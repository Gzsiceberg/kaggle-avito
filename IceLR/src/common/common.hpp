#ifndef _COMMON_H
#define _COMMON_H

#include <iostream>
#include <string>
#include <vector>
#include <cstdarg>
#include <boost/spirit/include/qi.hpp>
#include <boost/spirit/include/phoenix_operator.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/spirit/include/phoenix_core.hpp>
#include <boost/spirit/include/phoenix_operator.hpp>
#include <boost/spirit/include/phoenix_stl.hpp>
#include <boost/program_options.hpp>
#include <fstream>
#include "model.pb.h"
#include <memory> 
#include "timer.h"

int const kMaxLineSize = 1000000;
namespace qi = boost::spirit::qi;
namespace ascii = boost::spirit::ascii;
namespace phoenix = boost::phoenix;
namespace po = boost::program_options;

inline float logloss(float y, float p)
{
    p = fmax(fmin(p, 1.0 - 1e-6), 1e-6);
    return y > 0.5 ? -log(p) : -log(1.0 - p);
}

struct Error : std::exception
{
    char text[1000];

    Error(char const* fmt, ...) __attribute__((format(printf,2,3))) {
        va_list ap;
        va_start(ap, fmt);
        vsnprintf(text, sizeof text, fmt, ap);
        va_end(ap);
    }

    char const* what() const throw() { return text; }
};

FILE *open_c_file(std::string const &path, std::string const &mode)
{
    FILE *f = fopen(path.c_str(), mode.c_str());
    if(!f)
        throw std::runtime_error(std::string("cannot open ")+path);
    return f;
}

uint32_t get_nr_line(std::string const &path)
{
    FILE *f = open_c_file(path.c_str(), "r");
    char line[kMaxLineSize];

    uint32_t nr_line = 0;
    while(fgets(line, kMaxLineSize, f) != nullptr)
        ++nr_line;

    fclose(f);

    return nr_line;
}

template <typename Iterator>
bool parse_line(Iterator first, Iterator last, std::vector<double>& v)
{
    using qi::uint_;
    using qi::int_;
    using qi::float_;
    using qi::double_;
    using qi::_1;
    using qi::phrase_parse;
    using ascii::space;
    using phoenix::push_back;
    using phoenix::ref;

    bool r = phrase_parse(first, last,

        //  Begin grammar
        (
            *(double_[push_back(phoenix::ref(v), _1)]) 
        )
        ,
        //  End grammar
        space);

    if (first != last) // fail if we did not get a full match
        return false;
    return r;
}


template <typename Iterator>
bool parse_line(Iterator first, Iterator last, std::vector<uint32_t>& v)
{
    using qi::uint_;
    using qi::int_;
    using qi::float_;
    using qi::double_;
    using qi::_1;
    using qi::phrase_parse;
    using ascii::space;
    using phoenix::push_back;
    using phoenix::ref;

    bool r = phrase_parse(first, last,

        //  Begin grammar
        (
            *(uint_[push_back(phoenix::ref(v), _1)] | int_[push_back(phoenix::ref(v), _1)]) 
        )
        ,
        //  End grammar
        space);

    if (first != last) // fail if we did not get a full match
        return false;
    return r;
}

template <typename Iterator>
bool parse_line(Iterator first, Iterator last, float& y, std::vector<uint32_t>& v)
{
    using qi::uint_;
    using qi::int_;
    using qi::float_;
    using qi::double_;
    using qi::_1;
    using qi::phrase_parse;
    using ascii::space;
    using phoenix::push_back;
    using phoenix::ref;

    bool r = phrase_parse(first, last,

        //  Begin grammar
        (
            float_[ref(y) = _1] >> \
            *(uint_[push_back(phoenix::ref(v), _1)] | int_[push_back(phoenix::ref(v), _1)]) 
        )
        ,
        //  End grammar
        space);

    if (first != last) // fail if we did not get a full match
        return false;
    return r;
}

template <typename PROTO_OBJ>
void write_proto(const PROTO_OBJ& obj, std::string& buffer, std::ofstream& output)
{
    obj.SerializeToString(&buffer);
    size_t size = buffer.size();
    // printf("size: %d\n", size);
    output.write(reinterpret_cast<char*>(&size), sizeof(size));
    output.write(buffer.data(), size);
}

template <typename PROTO_OBJ>
bool read_proto(PROTO_OBJ& obj, char* buffer, size_t buffer_size, std::ifstream& input)
{
    size_t size;
    bool result = input.read(reinterpret_cast<char*>(&size), sizeof(size));
    if(!result)
        return false;
    // printf("size: %d buffer_size: %d\n", size, buffer_size);
    if(size <= 0 || size >= buffer_size || !input.read(buffer, size))
    {
        std::cerr << "Unexpected error while reading protobuf" << std::endl;
        return false;
    }
    obj.ParseFromArray(buffer, size);
    return true;
}

class MetaProblem
{
public:
    MetaProblem(){};
    ~MetaProblem(){};

    /* data */
    uint32_t prob_size;
};


class Problem
{
public:
    Problem(){};
    Problem(std::string& file_name, bool bias);
    ~Problem(){};

    void load_data(std::string& file_name, bool bias);

    void dump(std::string& path)
    {
        printf("dump file in %s\n", path.c_str());
        fflush(stdout);
        std::string buffer;
        std::ofstream output(path, std::ofstream::out | std::ofstream::trunc | std::ofstream::binary);

        MetaProb meta_prob;
        meta_prob.set_prob_size(meta_problem.prob_size);
        write_proto(meta_prob, buffer, output);

        size_t prob_size = all_y.size();
        for(size_t i = 0; i < prob_size; ++i)
        {
            float y = all_y[i];
            Instance instance;
            instance.set_y(y);
            std::vector<uint32_t>& f = all_f[i];
            for(auto& x : f) {
                instance.add_x(x);
            }
            write_proto(instance, buffer, output);
        }
        output.close();
    }

    void load(std::string& path)
    {
        printf("load file from %s\n", path.c_str());
        fflush(stdout);
        std::ifstream input(path, std::ifstream::in | std::ifstream::binary);
        if(!input)
        {
            throw Error(("load: cache path " + path + " doesn't exist").c_str());
        }

        Timer timer;
        std::unique_ptr<char[]> buffer(new char[kMaxLineSize]);
        MetaProb meta_prob;
        read_proto(meta_prob, buffer.get(), kMaxLineSize, input);
        meta_problem.prob_size = meta_prob.prob_size();
        all_y.resize(meta_problem.prob_size);
        all_f.resize(meta_problem.prob_size);

        Instance instance;
        uint32_t instance_count = 0;
        while(input && read_proto(instance, buffer.get(), kMaxLineSize, input))
        {
            all_y[instance_count] = instance.y();
            for(unsigned i = 0; i < instance.x_size(); ++i) {
                uint32_t x = instance.x(i);
                all_f[instance_count].push_back(x);
            }
            ++instance_count;
        }
        input.close();
        if(instance_count != meta_problem.prob_size)
            throw Error("cache file error.");
        printf("loading file done. time: %.2f\n", timer.toc());
    }

    /* data */
    std::vector< std::vector<uint32_t> > all_f;
    std::vector<float> all_y;
    MetaProblem meta_problem;
};

void Problem::load_data(std::string& file_name, bool bias=true)
{
    if(file_name.empty())
    {
        return;
    }
    std::string cache_file_name = file_name + ".cache";
    std::fstream _file(cache_file_name, std::ios::in);
    if(_file)
    {
        load(cache_file_name);
    }
    else
    {
        int line_count = get_nr_line(file_name);
        meta_problem.prob_size = line_count;
        all_y.resize(line_count, 0.0);
        all_f.resize(line_count);

        printf("start to parse file %s\n", file_name.c_str());
        Timer timer;
        FILE *f = open_c_file(file_name, "r");
        char line[kMaxLineSize];
        printf("%8s%9s%9s%10s\n", "iter", "time", "Y", "X_size");
        uint32_t print_iter = line_count/100 + 1;
        for(uint32_t i = 0; fgets(line, kMaxLineSize, f) != nullptr; ++i)
        {
            size_t length = strlen(line);
            if(length > 0)
            {
                bool r = parse_line(line, line + length, all_y[i], all_f[i]);
                if(!r)
                {
                    printf("failed to parse %d line.", i);
                }
                if(bias) all_f[i].push_back(1);
                if(i == print_iter)
                {
                    printf("%8d%9.1f%9.1f%10ld\n", i, timer.toc(), all_y[i], all_f[i].size());
                    print_iter *= 2;
                    fflush(stdout);
                }
            }
        }
        fclose(f);
        printf("parsing file done. time: %.2f\n", timer.toc());
        dump(cache_file_name);
    }
    _file.close();
    printf("problem size: %d\n", meta_problem.prob_size);
}
Problem::Problem(std::string& file_name, bool bias=true)
{
    load_data(file_name, bias);
}

class BaseEstimate
{
public:
    BaseEstimate(){}
    ~BaseEstimate(){}

    /* data */
    // virtual void fit(Problem &tr_prob, Problem &va_prob) = 0;
    // virtual float predict(std::vector<uint32_t> &f) = 0;
    // virtual void dump(std::string& path) = 0;
    // virtual void load(std::string& path) = 0;

};

#endif