# [Avito Context Ad Clicks](https://www.kaggle.com/c/avito-context-ad-clicks)

## System Requirement

- 64-bit Unix-like operating system 
- at least 20GB memory and 500GB disk space

## Dependencies and requirements

- Python 2.7
  
- pypy 2.1.0
  
- [XGBoost](https://github.com/dmlc/xgboost)
  
- [scons](http://www.scons.org/)
  
- numpy 1.9.2
  
- sklearn 0.15.2
  
- theano 0.7.0
  
- lasagne 0.1.dev0
  
- [nolearn](https://github.com/dnouri/nolearn)
  
- g++ (with C++11, OpenMp, [gflags](https://github.com/gflags/gflags) and [protobuf](https://github.com/google/protobuf))
  
  ​

## Instruction

- move all tsv files into ./data folder
  
- Compile the C++ code by 
  
- ``` sh
  cd IceLR
  scons
  ```
- run script to generate data and basic models
  
- ``` sh
  bash run.sh
  ```
- generate final submission by 
  
- ``` python
  pypy run.py --type ensemble --method nn > nn_log.log
  ```
