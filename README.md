# Gaussian Process Regression

This python application is an implementation of Gaussian process regression model, developed as a project for course "Fundamentals of Statistical Learning" at Arizona State University.

### This project was divided in three tasks:-

1. Find output for a given input vector by assuming arbitrary values for the gaussian parameters and report the accuracy.
2. Infer the gaussian parameters by minimizing negative log-likelihood of conditional distribution p(y|x)
3. Find output for given input vector by using the parameter values derived in 2nd task and report the change in accuracy.

### Brief description about the files present in this repository:-
1. gaussian_process.py - It is the python file to perform regression done in tasks 1 and 3
2. max_likelihood_estimator.py - This file is to infer the gaussian parameters by minimizing the cost function
3. data.mat - It is the given input mat file that contains input_train, target_train, input_val, target_val and input_test files
