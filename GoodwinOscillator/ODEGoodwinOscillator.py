# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 10:37:01 2018

@author: Mingjun
"""

import numpy as np
import pandas as pd
import scipy as sp
import argparse
import copy

# plot 3D surface
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

def get_arguments():
    parser = argparse.ArgumentParser(description='Markov chain Monte Carlo\
                                     methods for inferring parameters\
                                     in the Goodwin scillator ODE.\
                                     The distribution of them is highly multi-mode')
    parser.add_argument('--ode_parameters', \
                        default=[10.0,1.0,3.0,0.5,2.0,1.0,1.0],
                        help='the parameters for Goodwin scillator ODE\
                        in the order: rho,a1,a2,alpha,k1,k2,...')
    parser.add_argument('--nosOfS',
                        type=int,
                        default=81,
                        help='Number of steps for updating ode')
    parser.add_argument('--deltaS',
                        type=float,
                        default=1.0,
                        help='Step size for descretizing ODE')
    parser.add_argument('--sigma',
                        type=float,
                        default=0.1,
                        help='Noise deviation of likelihood model')
    parser.add_argument('-target_param_label', default=[4,5], \
                        help='list of labels of parameters to simulate', \
                        type=int)
    parser.add_argument('--startTimeToObserv',
                        type=int,
                        default=40,
                        help='The start time point to generate the observations')
    parser.add_argument('--nosOfObserv',
                        type=int,
                        default=2,
                        help='Number of latent variable assumed to \
                        generate observations')
    parser.add_argument('--gamma_shape',
                        type=float,
                        default=2.0,
                        help='Shape parameter of Gamma distribution')    
    parser.add_argument('--gamma_rate',
                        type=float,
                        default=1.0,
                        help='Rate parameter of Gamma distribution')
    parser.add_argument('--xlim',
                        type=float,
                        default=5.0,
                        help='x-axis limit')
    parser.add_argument('--ylim',
                        type=float,
                        default=10.0,
                        help='y-axis limit')
    parser.add_argument('--nosOfChains',
                        type=int,
                        default=50,
                        help='Number of chains for parallel tempering')
    parser.add_argument('--power',
                        type=int,
                        default=5.0,
                        help='The powers')
    return parser.parse_args()

def discretize_ode(parameters,x0,nosOfS,deltaS):
    # ##########The discretized Goodwin oscillator ODE ########################
    # x1 represents the concentration of mRNA for a target gene;
    # x2 reresents the corresponding protein product of the gene
    # x3,...,x_g represent intermediate protein species that facilitate 
    #    a cascade of enzymatic activation that ultimately leads to 
    #    a negative feedback, via xg, on the rate at which mRNA is transcribed.
    # Given the parameters, this function generates trajectories of x1,..x_g.
    
    ########### parameters ####################################################
    # x0: start point
    # a1, a2, alpha, rho: parameters, rho>8
    # kappa: a vector parameter; g=len(kappa)+1: number of species.
    ###########################################################################
    
    ## read the parameters
    rho = parameters[0]
    a1 = parameters[1]
    a2 = parameters[2]
    alpha = parameters[3]
    kappa = copy.copy(parameters[4:])
    
    # the number of species
    g = len(x0)
    
    # represent x by using numpy array
    x = np.array(x0).flatten().reshape((1,g))
    
    # iteratively generate the states of x
    for s in range(nosOfS):
        #print('number of iterations:{0}'.format(s))
        x_old = x[-1]
        x_new = x_old[0] + deltaS*(a1/(1.0+a2*np.power(x_old[-1],rho))\
                     -alpha*x_old[0])
        #print(x_new)
        for j in range(1,g):
            xj = x_old[j] + deltaS*(kappa[j-1]*x_old[j-1] - alpha*x_old[j])
            x_new = np.append(x_new,xj)
        #print(x_new)
        x = np.append(x,np.array(x_new).reshape((1,g)),axis=0)
    return x

def plot_trajectory(x):
    pd.DataFrame(x).plot()

def generate_observations(x,sigma):
    # observations are generated by independent Gaussian
    # y ~ Normal(x,sigma^2)
    observation = np.random.normal(x,sigma)
    return observation

def plot_trajectory_points(x,observ):
    ax = pd.DataFrame(x).plot()
    pd.DataFrame(observ).plot(ax=ax,marker='o')
    
def lognormal(y,x,sigma):
    # Compute the log Normal distribution element-wisely. x,y are arrays
    # x: the mean
    # y: the Normal random variable
    m,n=np.shape(y)
    deltaYX2 = np.power(y-x,2)
    logy = -0.5*m*n*np.log(2*np.pi*np.power(sigma,2))\
            -0.5*(1/np.power(sigma,2))*np.sum(deltaYX2)
    return logy
    
def loglikelihood(observ,parameters,x0,nosOfS,deltaS,nosOfObserv,\
                  sigma,startTimeToObserv,temper):
    # To compute likelihood, we need to compute the ODE trajectory.  
    # The observed data must be corresponding to `x'.
    # The ODE trajectory is computed given these parameters.
    # startTimeToObserv: the assumed time to observe the data
    # Note: x is a S*d matrix where S is the time steps of discretization, and
    #       d is the number of latent variables.
    # temper: the inverse temperature to represent the power posterior 
    # We assume only `a' number of variables - nosOfObserv - are observed.
    
    ## read the parameters
    rho = parameters[0]
    a1 = parameters[1]
    a2 = parameters[2]
    alpha = parameters[3]
    kappa = copy.copy(parameters[4:])
    
    x = discretize_ode(parameters,x0,nosOfS,deltaS)
    xobserved = x[startTimeToObserv:,0:nosOfObserv]
    observ = observ[startTimeToObserv:,0:nosOfObserv]
    logy = temper*lognormal(observ,xobserved,sigma)
    return logy

def loggamma(x,alpha,beta):
    # Compute log Gamma distribution given parameters shape alpha and rate beta
    # x is positive
    logx = alpha*np.log(beta) + (alpha-1.0)*np.log(x) - beta*np.array(x) \
            - np.log(sp.special.gamma(alpha))
    return np.sum(logx)

def listflatten(alist):
    # flatten sublist in list
    return [item for sublist in alist for item in sublist]

def conditionalPosterior(observ,x0,parameters,temper,\
                         nosOfS=81,deltaS=1.0,gamma_shape=2.0,gamma_rate=1.0,\
                         nosOfObserv=2,sigma=0.1,startTimeToObserv=40):
    # this function only calculate the conditional posterior of two parameters
    # I should write a clever code to vary these parameters
    # currently we only vary a1 and a2
    
    ## read the parameters
    rho = parameters[0]
    a1 = parameters[1]
    a2 = parameters[2]
    alpha = parameters[3]
    kappa = copy.copy(parameters[4:])
    
    kappa = np.array(kappa)
    
    # log likelihood
    logll = loglikelihood(observ,parameters,x0,nosOfS,deltaS,nosOfObserv,\
                  sigma,startTimeToObserv,temper)
    
    # log prior
    parameters_gamma = []
    for index in args.target_param_label:
        parameters_gamma.append(parameters[index])
    logprior = loggamma(parameters_gamma,gamma_shape,gamma_rate)
    # compute the conditional posterior
    return logll+logprior

def surface_conditionalPosterior(observ,x0,parameters,temper,\
                         nosOfS=81,deltaS=1.0,gamma_shape=2.0,gamma_rate=1.0,\
                         nosOfObserv=2,sigma=0.1,startTimeToObserv=40):
    ## read the parameters
    rho = parameters[0]
    a1 = parameters[1]
    a2 = parameters[2]
    alpha = parameters[3]
    kappa = copy.copy(parameters[4:])
    
    # compute log(p(a1,a2|....)) whene varying a1 and a2
    a1_list = list(np.arange(0.001,args.xlim,0.1))
    a2_list = list(np.arange(0.001,args.ylim,0.1))
    a1_grid = np.zeros((len(a2_list),len(a1_list)))
    a2_grid = np.zeros((len(a2_list),len(a1_list)))
    y_grid = np.zeros((len(a2_list),len(a1_list)))
    for i,a1 in enumerate(a1_list):
        for j,a2 in enumerate(a2_list):
            a1_grid[j][i] = a1
            a2_grid[j][i] = a2
            parameters[args.target_param_label[0]] = a1
            parameters[args.target_param_label[1]] = a2
            logpost = conditionalPosterior(observ,x0,parameters,temper,\
                         nosOfS=nosOfS,deltaS=deltaS,\
                         gamma_shape=gamma_shape,gamma_rate=gamma_rate,\
                         nosOfObserv=nosOfObserv,sigma=sigma,\
                         startTimeToObserv=startTimeToObserv)
            y_grid[j][i] = logpost
    return a1_grid,a2_grid,y_grid

def plot_surface(a1_grid,a2_grid,y_grid):
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # Plot the surface.
    surf = ax.plot_surface(a1_grid,a2_grid,y_grid, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    
    # Customize the z axis.
    #ax.set_zlim(-1.01, 1.01)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    
    plt.show()
    
def plot_contour(X,Y,Z):
    plt.figure()
    N = 200
    CS = plt.contour(X, Y, Z, N)
    plt.clabel(CS, inline=1, fontsize=10)
    
    plt.figure()
    cs = plt.contourf(X, Y, Z)
    #fig.colorbar(cs, ax=axs[0], format="%.2f")
    
def plot_trajectory(x,y,XX,YY,ZZ):
    plt.figure(1)
    N = 1000
    cs = plt.contour(XX, YY, ZZ, N)
    plt.clabel(cs, inline=1, fontsize=10)
    
    plt.plot(x,y,'r',linewidth=3)
    plt.plot(x[0],y[0],'ko',markersize=10)
    plt.plot(x[-1],y[-1],'b*',markersize=10)
    amax = np.max([np.max(x),np.max(y)])
    #plt.xlim(0.0,0.01+amax)
    #plt.ylim(0.0,0.01+amax)

def mh(observ,x0,parameters,temper,nosOfS=81,deltaS=1.0,\
       gamma_shape=2.0,gamma_rate=1.0,\
       nosOfObserv=2,sigma=0.1,startTimeToObserv=40,\
       nosOfSamples=1000,stepsize=0.05):
    # We only want to simulate two parameters: 
    #    args.target_param1 and args.target_param2
    
    # initializee parameters
#    parameters[args.target_param1] = np.random.gamma(gamma_shape,gamma_rate)
#    parameters[args.target_param2] = np.random.gamma(gamma_shape,gamma_rate)
    
    # labels of parameters to be updated
    target_labels = copy.copy(args.target_param_label)
    
    # store the initial parameters
    old_params = copy.copy(parameters)
    
    # stepsize
    #stepsize = 0.05
    
    # the old log posterior
    logpost_old = conditionalPosterior(observ,x0,old_params,temper,\
                         nosOfS=nosOfS,deltaS=deltaS,\
                         gamma_shape=gamma_shape,gamma_rate=gamma_rate,\
                         nosOfObserv=nosOfObserv,sigma=sigma,\
                         startTimeToObserv=startTimeToObserv)
    for isamp in range(nosOfSamples):
        # propose a new state for these parameters
        #print(old_params)
        
        proposed_state = copy.copy(old_params)
        jacobian_old = 0.0
        jacobian_new = 0.0
        #print(old_params)
        for label in target_labels:
            # Note: we use a function to define the proposals because the 
            # parameters have to be positive so param=exp(epsilon) 
            # where we instead draw samples for epsilon~q(epsilon_new|epsilon_old)
            # where epsilon=log(param). So epsilon is a symmetric Gaussian which
            # could be cancelled when computing acceptance probability, 
            # but we still need to compute Jacobian=1/|param|
            # Therefore r=log(p(param_new))-log(p(param_old))+log(1/|param_old|)
            #             -log(1/|param_new|)
            epsilon = np.random.normal(loc=np.log(old_params[label]),\
                                       scale=stepsize)
            proposed_state[label] = np.exp(epsilon)
            jacobian_old = jacobian_old - np.log(np.abs(old_params[label]))
            jacobian_new = jacobian_new - np.log(np.abs(proposed_state[label]))
            
        logpost_new = conditionalPosterior(observ,x0,proposed_state,temper,\
                         nosOfS=nosOfS,deltaS=deltaS,\
                         gamma_shape=gamma_shape,gamma_rate=gamma_rate,\
                         nosOfObserv=nosOfObserv,sigma=sigma,\
                         startTimeToObserv=startTimeToObserv)
        # note the proposal is Gaussian which symmetric, so its cancelled
        prob = np.min([1.0,\
                       logpost_new - logpost_old + jacobian_old - jacobian_new])
        u = np.log(np.random.uniform())
        if u<prob:
            # accept the proposal
            old_params = copy.copy(proposed_state)
            logpost_old = copy.copy(logpost_new)
            #print('old:{0}'.format(old_params))
        #print('oldold:{0}'.format(old_params))
        if isamp==0:
            samples = np.array(listflatten([list(old_params)]))\
                        .reshape((1,len(old_params)))
        else:
            samples = np.append(samples,
                        np.array(listflatten([list(old_params)]))\
                        .reshape((1,len(old_params))),
                        axis = 0)
        print('log_accept_prob:{0}'.format(prob))
    return samples

def mh_onestep(observ,x0,parameters,temper,nosOfS=81,deltaS=1.0,\
       gamma_shape=2.0,gamma_rate=1.0,\
       nosOfObserv=2,sigma=0.1,startTimeToObserv=40,stepsize=0.05):
    # We only want to simulate two parameters: 
    #    args.target_param1 and args.target_param2
    
    # This function only move one step
    
    # labels of parameters to be updated
    target_labels = copy.copy(args.target_param_label)
    
    # the old state
    old_params = copy.copy(parameters)
    
    # stepsize
    #stepsize = 0.05
    
    # the old log posterior
    logpost_old = conditionalPosterior(observ,x0,old_params,temper,\
                         nosOfS=nosOfS,deltaS=deltaS,\
                         gamma_shape=gamma_shape,gamma_rate=gamma_rate,\
                         nosOfObserv=nosOfObserv,sigma=sigma,\
                         startTimeToObserv=startTimeToObserv)
    
    # now propose a new state
    proposed_state = copy.copy(old_params)
    jacobian_old = 0.0
    jacobian_new = 0.0
    #print(old_params)
    for label in target_labels:
        # Note: we use a function to define the proposals because the 
        # parameters have to be positive so param=exp(epsilon) 
        # where we instead draw samples for epsilon~q(epsilon_new|epsilon_old)
        # where epsilon=log(param). So epsilon is a symmetric Gaussian which
        # could be cancelled when computing acceptance probability, 
        # but we still need to compute Jacobian=1/|param|
        # Therefore r=log(p(param_new))-log(p(param_old))+log(1/|param_old|)
        #             -log(1/|param_new|)
        epsilon = np.random.normal(loc=np.log(old_params[label]),\
                                   scale=stepsize)
        proposed_state[label] = np.exp(epsilon)
        jacobian_old = jacobian_old - np.log(np.abs(old_params[label]))
        jacobian_new = jacobian_new - np.log(np.abs(proposed_state[label]))
        
    logpost_new = conditionalPosterior(observ,x0,proposed_state,temper,\
                     nosOfS=nosOfS,deltaS=deltaS,\
                     gamma_shape=gamma_shape,gamma_rate=gamma_rate,\
                     nosOfObserv=nosOfObserv,sigma=sigma,\
                     startTimeToObserv=startTimeToObserv)
    # note the proposal is Gaussian which symmetric, so its cancelled
    prob = np.min([1.0,\
                   logpost_new - logpost_old + jacobian_old - jacobian_new])
    u = np.log(np.random.uniform())
    if u<prob:
        # accept the proposal
        old_params = copy.copy(proposed_state)
        logpost_old = copy.copy(logpost_new)
        old_params[-1] = copy.copy(logpost_new)
    #print('log_accept_prob:{0}'.format(prob))
#    asample = np.array(listflatten([old_params,[logpost_old]]))#.\
                #reshape((1,1+len(old_params)))
    return old_params

### Parallel tempering
def parallel_tempering(observ,x0,nosOfS=81,deltaS=1.0,\
       gamma_shape=2.0,gamma_rate=1.0,\
       nosOfObserv=2,sigma=0.1,startTimeToObserv=40,\
       nosOfSamples=1000,stepsize=0.05):
    
    # The population of samples
    numberOfChains = args.nosOfChains
    populationOfSamples = {}
    
    # The powers
    powers = np.power(np.arange(0,numberOfChains+1,1.0)/numberOfChains,\
                      args.power)
    
    # the labels for those parameters to simulate
    target_labels = copy.copy(args.target_param_label)
    
    # Initialise the population parameters    
    parameters = copy.copy(np.array(listflatten([args.ode_parameters,[0.0]])))
    for temper in powers:        
        for label in target_labels:
            parameters[label] = copy.copy(np.random.gamma(gamma_shape,\
                                          gamma_rate))
            if temper==1.0:
                parameters[label] = 4.0
        populationOfSamples[temper] = copy.copy(parameters.\
                                        reshape((1,len(parameters))))
        
    # update each chain in parallel and also swap chains
    for nsamp in range(nosOfSamples):
        print('nos of samples: {0}'.format(nsamp))
        probswitch = np.random.uniform()
        if probswitch < 0.5:
            # parallel step: update each chain
            for ichain,temper in enumerate(powers):
                #print('temper:{0}'.format(temper))
                parameters = copy.copy(populationOfSamples[temper][-1,:])
                sample = mh_onestep(observ,x0,parameters,temper,\
                             nosOfS=nosOfS,deltaS=deltaS,\
                             gamma_shape=gamma_shape,gamma_rate=gamma_rate,\
                             nosOfObserv=nosOfObserv,sigma=sigma,\
                             startTimeToObserv=startTimeToObserv,\
                             stepsize=stepsize)
                asample = copy.copy(populationOfSamples[temper])
                asample = np.append(asample,sample.\
                                    reshape((1,len(sample))),axis=0)
                populationOfSamples[temper] = copy.copy(asample)
        else:
            # swap neighbor chains
            # select a neighbour pair
            ichain = np.random.choice(np.arange(0,numberOfChains))
            
            # a sample of the i^th chain
            sample_ichain = populationOfSamples[powers[ichain]][-1,:]
            param_i = copy.copy(sample_ichain[0:-1])
            logpost_i = copy.copy(sample_ichain[-1:])
            
            # a sample of the j=i+1 chain
            sample_jchain = populationOfSamples[powers[ichain+1]][-1,:]
            param_j = copy.copy(sample_jchain[0:-1])
            logpost_j = copy.copy(sample_jchain[-1:])
            
            # we want to swap chain i and j
            logpost_ij=conditionalPosterior(observ,x0,param_j,\
                                            powers[ichain],\
                     nosOfS=nosOfS,deltaS=deltaS,\
                     gamma_shape=gamma_shape,gamma_rate=gamma_rate,\
                     nosOfObserv=nosOfObserv,sigma=sigma,\
                     startTimeToObserv=startTimeToObserv)
            logpost_ji=conditionalPosterior(observ,x0,param_i,\
                                            powers[ichain+1],\
                     nosOfS=nosOfS,deltaS=deltaS,\
                     gamma_shape=gamma_shape,gamma_rate=gamma_rate,\
                     nosOfObserv=nosOfObserv,sigma=sigma,\
                     startTimeToObserv=startTimeToObserv)
            
            #compute the swap probability
            r = logpost_ij + logpost_ji - logpost_i - logpost_j
            prob = np.min([1.0, r])
            u = np.log(np.random.uniform())
            if u<prob:
                # accept to swap chains
                populationOfSamples[powers[ichain]][-1,:] = \
                np.array(listflatten([list(param_j),[logpost_ij]]))
                populationOfSamples[powers[ichain+1]][-1,:] = \
                np.array(listflatten([list(param_i),[logpost_ji]]))
    return populationOfSamples

################################################################
# model parameters
a1 = 1.0
a2 = 3.0
alpha = 0.5
rho = 10
g = 3
kappa = []
for i in range(g-1):
    if i==0:
        kappa.append(2.0)
    else:
        kappa.append(1.0)
kappa = np.array(kappa)

# setup parameters
args = get_arguments()
parameters = [[rho],[a1],[a2],[alpha],kappa.tolist()]
args.ode_parameters = copy.copy(np.array(listflatten(parameters)))
parameters = copy.copy(args.ode_parameters)
sigma = args.sigma
nosOfS = args.nosOfS
deltaS = args.deltaS
nosOfObserv = args.nosOfObserv
startTimeToObserv = args.startTimeToObserv

gamma_shape = args.gamma_shape
gamma_rate = args.gamma_rate

# starting point
x0 = np.zeros(g)

# inverse temperature
temper = 1.0

############# do something from here ################
x = discretize_ode(parameters,x0,nosOfS,deltaS)
#plot_trajectory(x)

# generate some observations for simulating
observ = generate_observations(x,sigma)


#plot_trajectory_points(x,observ[:,0:nosOfObserv])
#
#logy = loglikelihood(observ,parameters,x0,nosOfS,deltaS,\
#                     nosOfObserv,sigma,startTimeToObserv)
#
#logx = loggamma(parameters,gamma_shape,gamma_rate)
#
#logpost = conditionalPosterior(observ,x0,parameters,temper,\
#                         nosOfS=nosOfS,deltaS=deltaS,gamma_shape=gamma_shape,\
#                         gamma_rate=gamma_rate,\
#                         nosOfObserv=nosOfObserv,sigma=sigma,\
#                         startTimeToObserv=startTimeToObserv)
#
a1_grid,a2_grid,y_grid = surface_conditionalPosterior(observ,x0,parameters,\
                         temper,\
                         nosOfS=nosOfS,deltaS=deltaS,\
                         gamma_shape=gamma_shape,gamma_rate=gamma_rate,\
                         nosOfObserv=nosOfObserv,sigma=sigma,\
                         startTimeToObserv=startTimeToObserv)

#plot_surface(a1_grid,a2_grid,y_grid)

#plot_contour(a1_grid,a2_grid,y_grid)

# Use a MH algorithm to simulate these parameters
# labels of parameters to be updated

#target_labels = copy.copy(args.target_param_label)
#for label in target_labels:
#    parameters[label] = np.random.gamma(gamma_shape,gamma_rate)
#parameters[target_labels[0]] = 4.0
#parameters[target_labels[1]] = 4.0
#samples = mh(observ,x0,parameters,temper,nosOfS=81,deltaS=1.0,\
#                         gamma_shape=2.0,gamma_rate=1.0,\
#                         nosOfObserv=2,sigma=0.1,startTimeToObserv=40,\
#                         nosOfSamples=20000,stepsize=0.05)
#x = samples[:,target_labels[0]]
#y = samples[:,target_labels[1]]
#plot_trajectory(x,y,a1_grid,a2_grid,y_grid)

# parallel tempering 
populationOfSample = parallel_tempering(observ,x0,\
                                        nosOfS=nosOfS,deltaS=deltaS,\
                                        gamma_shape=gamma_shape,\
                                        gamma_rate=gamma_rate,\
                                        nosOfObserv=nosOfObserv,\
                                        sigma=sigma,\
                                        startTimeToObserv=startTimeToObserv,\
                                        nosOfSamples=100,stepsize=0.05)

## plot all the population trajectories
#for nsamp in populationOfSample:
#    samples = copy.copy(populationOfSample[nsamp])
#    x = copy.copy(samples[:,target_labels[0]])
#    y = copy.copy(samples[:,target_labels[1]])
#    plot_trajectory(x,y,a1_grid,a2_grid,y_grid)
    
# only plot the trajectory of target distribution
samples = copy.copy(populationOfSample[1.0])
x = copy.copy(samples[:,target_labels[0]])
y = copy.copy(samples[:,target_labels[1]])
plot_trajectory(x,y,a1_grid,a2_grid,y_grid)