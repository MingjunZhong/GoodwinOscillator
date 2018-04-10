# Bayesian inference for Goodwin model of oscillatory enzymatic control

The model is defined by Goodwin (1965). The model is nonlinear ODE. We attempt to perform Bayesian inference over nonlinear ODE model parameters. Calderhead and Girolami (2009) studied this model using Bayesian inference. 

This code is to employ Metropolis-Hastings algorithms and Parallel Tempering technique to simulate the posterior of the ODE model parameters.

The parameters to be simulated: \alpha, a_1, a_2, k_1, k_2, ...
