# Bayesian Optimization (BO) with Expected Improvement.

## Introduction
The code implements a Bayesian optimizer with Gaussian process for tuning hyper-parameters. Expected Improvement is used as the standard when choosing the next point for evaluation. The implementation is in bo.py. 

As an example, the optimizer is used for tuning parameters of a random forest classifier, which is then used for classifying digits in the MNIST dataset.

The code was completed in the lab of the Advanced Machine Learning course (2013/14) in the University of Oxford.

Package required: numpy, scipy, sklearn

For how to use the optimizer, please refer to tune_random_forests.py.
To run the demo, simply do
```bash
python tune_random_forests.py
```
