# %% [markdown]
# Top
# 

# %%

# %%
# Value Iteration for PIMDP Reachability Checking
# Jesse Jiang

# Import classes
import numpy as np
# Import Aziz Phase Space Planning Helper
from psp import *
import csv
# Define state parameters
gridSize = 10
stateLength = 0.15
footLength = 0.3
angleSize = 15
angleLength = 360/angleSize
numStates = int(gridSize ** 2 * angleLength)*2
footSpace = 0.1
stepLength = 0.3
maxSteps = 3

# Class for PIMDP state
class State:

  # Constructor Method
  def __init__(self, id, actions, lower, upper, stateType, counter):
    # Set state ID
    self.id = id
    # Possible actions
    self.actions = actions
    self.targets = [0]*len(actions)
    # Lower transition bounds for each action
    # Dictionary format for each action, {nextState:transitionProb}
    self.lower = [lower[x] for x in range(len(actions))]
    # Upper transition bounds for each action
    self.upper = [upper[x] for x in range(len(actions))]
    # Accepting = 'a', Rejecting = 'r', Uncertain = 'u'
    self.stateType = stateType
    # Initial probability of satisfying specification
    # 1 if accepting, 0 if rejecting or uncertain
    if stateType == 'a':
      self.psat = 1
    else:
      self.psat = 0
    # State parameters
    self.center = []
    # GP Parameters
    self.zminState = 0
    self.zmaxState = 0
    self.zUncertaintyState = 0
    self.dmin = [0]*len(actions)
    self.dmax = [0]*len(actions)
    self.zmin = [0]*len(actions)
    self.zmax = [0]*len(actions)
    self.zave = [0]*len(actions)
    self.thetamin = [0]*len(actions)
    self.thetamax = [0]*len(actions)
    self.zUncertainty = [np.array([[0]])]*len(actions)
    self.counter = counter

  # Set state id
  def setID(self,id):
    self.id = id

  # Return state id
  def getID(self):
    return self.id

  # Return counter
  def getCounter(self):
    return self.counter

  # Set counter
  def setCounter(self,counter):
    self.counter = counter
  
  # Append a new action
  def append(self, action, lower, upper):
    self.actions.append(action)
    self.lower.append(lower)
    self.upper.append(upper)

  # Remove an action
  def remove(self, action):
    idx = self.actions.index(action)
    self.actions.pop(idx)
    self.lower.pop(idx)
    self.upper.pop(idx)
    self.targets.pop(idx)
    self.dmin.pop(idx)
    self.dmax.pop(idx)
    self.zmin.pop(idx)
    self.zmax.pop(idx)
    self.thetamin.pop(idx)
    self.thetamax.pop(idx)

  # Return probability of satisfying specification
  def getPSat(self):
    return self.psat

  # Return set of actions
  def getActions(self):
    return self.actions

  # Return lower transition probabilities for an action
  def getLower(self, action):
    idx = self.actions.index(action)
    return self.lower[idx]

  # Return upper transition probabilities for an action
  def getUpper(self, action):
    idx = self.actions.index(action)
    return self.upper[idx]
  
  # Return next states for a given action
  def getNext(self, action):
    idx = self.actions.index(action)
    return self.lower[idx].keys()

  # Set state type
  def setStateType(self, stateType):
    self.stateType = stateType

  # Update transition probability
  def update(self, action, lower, upper):
    idx = self.actions.index(action)
    self.lower[idx] = lower
    self.upper[idx] = upper

  # Set probability of satisfying specification
  def setPSat(self, psat):
    self.psat = psat

  # Set optimal control action
  def setOptAct(self, action):
    self.optimalAction = action

  # Set center point
  def setCenter(self,center):
    self.center = center

  # Set target region for each action
  def setTarget(self,action,target):
    idx = self.actions.index(action)
    self.targets[idx]=target

  # Get state type
  def getStateType(self):
    return self.stateType

  # Print info
  def getInfo(self):
    print("State ID: " + str(self.id))
    for action in self.actions:
      idx = self.actions.index(action)
      print("Action " + action + "\nLower Transitions: ")
      print(self.lower[idx])
      print("Upper Transitions: ")
      print(self.upper[idx])
      print("\n")

  # Get optimal control action
  def getOptAct(self):
    return self.optimalAction

  # Get target region for each action
  def getTarget(self,action):
    idx = self.actions.index(action)
    return self.targets[idx]

  # Get center for state
  def getCenter(self):
    return self.center

# Class for PIMDP State Space
class StateSpace:

  # Constructor Method
  def __init__(self, states):
    # Initialize state
    self.states = states
    # Initialize state IDs
    self.ids = []
    for state in states:
      self.ids.append(state.getID())
    # Initialize set of all actions
    self.actions = []
    self.lowers = []
    self.uppers = []
    self.psats = []
    self.minOrder = 0
    self.maxOrder = 0

  # Append State
  def append(self, state):
    self.states.append(state)
    self.ids.append(state.getID())

  # Remove State
  def remove(self, state):
    self.states.remove(state)
    self.ids.remove(state.getID())

  # Find all feasible actions
  def findActions(self):
    # Loop through all states
    self.actions=[]
    for state in self.states:
      # get actions
      tempActions = state.getActions()
      for action in tempActions:
        # append new actions
        if action not in self.actions:
          self.actions.append(action)

  # Find number of uncertain states
  def countUncertain(self):
    self.numUncertain = 0
    for state in self.states:
      if state.getStateType() == 'u':
        self.numUncertain = self.numUncertain + 1

  # Construct Lower and Upper Transition Probability Matrix
  def constructTransitions(self):
    self.lowers = []
    self.uppers = []
    # Create one transition matrix for each action
    for action in self.actions:
      tempLower = np.zeros([len(self.states),len(self.states)])
      tempUpper = np.zeros([len(self.states),len(self.states)])
      # Loop through states
      for state in self.states:
        id = state.getID()
        # Append transition probabilities if applicable
        if action in state.getActions():
            next = state.getNext(action)
            lower = state.getLower(action)
            upper = state.getUpper(action)
            for ns in next:
              tempLower[ns,id] = lower.get(ns)
              tempUpper[ns,id] = upper.get(ns)
      self.lowers.append(tempLower)
      self.uppers.append(tempUpper)

  # Construct vector of PSats
  def getPSats(self):
    self.psats=np.empty([len(self.states),1])
    # Loop through states and append PSats
    for state in range(len(self.states)):
      self.psats[state] = self.states[state].getPSat()

  # Construct list of optimal actions
  def getOptActs(self):
    self.optActs=[]
    # Loop through states and append PSats
    for state in self.states:
      self.optActs.append(state.getOptAct())

  # Order states
  def orderStates(self):
    self.getPSats()
    # Return states ordered from lowest PSat to highest
    return np.argsort(self.psats,axis=0)

  # Construct minimizing transition matrix based on state ordering
  def orderMinTransitions(self):
    self.orderedTransitions = []
    # Sort states from lowest to highest PSat
    self.order = self.orderStates()
    self.minOrder = self.order
    # Loop through each transition matrix
    for act in range(len(self.actions)):
      lower = self.lowers[act]
      upper = self.uppers[act]
      tempMat = np.zeros([len(self.states),len(self.states)])
      # Loop through each row
      for col in range(len(self.states)):
        # Loop through each potential ordering
        self.remainder = 0
        # Identify potential transition states
        if not self.actions[act] in self.states[col].getActions():
          tempMat[:,col]=0
        else:
          ids = np.fromiter(self.states[col].getNext(self.actions[act]),dtype=int)
          orig_indices = np.squeeze(self.order.argsort(axis=0))
          ndx = orig_indices[np.searchsorted(np.squeeze(self.order[orig_indices]), ids)]
          for i in np.sort(ndx):
            self.remainder = 0
            # Add appropriate lower and upper probabilities
            probs = np.sum(upper[self.order[0:i],col]) + np.sum(lower[self.order[i+1:],col])
            # Check if remainder is a valid probability
            diff = 1-probs
            if ((diff<=upper[self.order[i],col]) and (diff>=lower[self.order[i],col])):
              self.remainder = diff
              break
          # Assign final transition probabilities
          tempMat[self.order[0:i],col] = upper[self.order[0:i],col]
          tempMat[self.order[i+1:],col] = lower[self.order[i+1:],col]
          tempMat[self.order[i],col] = self.remainder
      # Construct final matrix for given action
      self.orderedTransitions.append(tempMat)

  # Construct minimizing transition matrix based on predetermined state ordering
  def orderMinTransitionsSimple(self,order):
    self.orderedTransitions = []
    # Sort states from lowest to highest PSat
    self.order = order
    # Loop through each transition matrix
    for act in range(len(self.actions)):
      lower = self.lowers[act]
      upper = self.uppers[act]
      tempMat = np.zeros([len(self.states),len(self.states)])
      # Loop through each row
      for col in range(len(self.states)):
        # Loop through each potential ordering
        self.remainder = 0
        # Identify potential transition states
        if not self.actions[act] in self.states[col].getActions():
          tempMat[:,col]=0
        else:
          ids = np.fromiter(self.states[col].getNext(self.actions[act]),dtype=int)
          orig_indices = np.squeeze(self.order.argsort(axis=0))
          ndx = orig_indices[np.searchsorted(np.squeeze(self.order[orig_indices]), ids)]
          for i in np.sort(ndx):
            self.remainder = 0
            # Add appropriate lower and upper probabilities
            probs = np.sum(upper[self.order[0:i],col]) + np.sum(lower[self.order[i+1:],col])
            # Check if remainder is a valid probability
            diff = 1-probs
            if ((diff<=upper[self.order[i],col]) and (diff>=lower[self.order[i],col])):
              self.remainder = diff
              break
          # Assign final transition probabilities
          tempMat[self.order[0:i],col] = upper[self.order[0:i],col]
          tempMat[self.order[i+1:],col] = lower[self.order[i+1:],col]
          tempMat[self.order[i],col] = self.remainder
      # Construct final matrix for given action
      self.orderedTransitions.append(tempMat)

  # Construct maximizing transition matrix based on state ordering
  def orderMaxTransitions(self):
    self.orderedTransitions = []
    # Sort states from lowest to highest PSat
    self.order = self.orderStates()
    self.order = self.order[::-1]
    self.maxOrder = self.order
    # Loop through each transition matrix
    for act in range(len(self.actions)):
      lower = self.lowers[act]
      upper = self.uppers[act]
      tempMat = np.zeros([len(self.states),len(self.states)])
      # Loop through each row
      for col in range(len(self.states)):
        # Loop through each potential ordering
        self.remainder = 0
        # Identify potential transition states
        if not self.actions[act] in self.states[col].getActions():
          tempMat[:,col]=0
        else:
          ids = np.fromiter(self.states[col].getNext(self.actions[act]),dtype=int)
          orig_indices = np.squeeze(self.order.argsort(axis=0))
          ndx = orig_indices[np.searchsorted(np.squeeze(self.order[orig_indices]), ids)]
          for i in np.sort(ndx):
            self.remainder = 0
            # Add appropriate lower and upper probabilities
            probs = np.sum(upper[self.order[0:i],col]) + np.sum(lower[self.order[i+1:],col])
            # Check if remainder is a valid probability
            diff = 1-probs
            if ((diff<=upper[self.order[i],col]) and (diff>=lower[self.order[i],col])):
              self.remainder = diff
              break
          # Assign final transition probabilities
          tempMat[self.order[0:i],col] = upper[self.order[0:i],col]
          tempMat[self.order[i+1:],col] = lower[self.order[i+1:],col]
          tempMat[self.order[i],col] = self.remainder
      # Construct final matrix for given action
      self.orderedTransitions.append(tempMat)

  # Construct maximizing transition matrix based on predetermined state ordering
  def orderMaxTransitionsSimple(self,order):
    self.orderedTransitions = []
    # Sort states from lowest to highest PSat
    self.order = order
    # Loop through each transition matrix
    for act in range(len(self.actions)):
      lower = self.lowers[act]
      upper = self.uppers[act]
      tempMat = np.zeros([len(self.states),len(self.states)])
      # Loop through each row
      for col in range(len(self.states)):
        # Loop through each potential ordering
        self.remainder = 0
        # Identify potential transition states
        if not self.actions[act] in self.states[col].getActions():
          tempMat[:,col]=0
        else:
          ids = np.fromiter(self.states[col].getNext(self.actions[act]),dtype=int)
          orig_indices = np.squeeze(self.order.argsort(axis=0))
          ndx = orig_indices[np.searchsorted(np.squeeze(self.order[orig_indices]), ids)]
          for i in np.sort(ndx):
            self.remainder = 0
            # Add appropriate lower and upper probabilities
            probs = np.sum(upper[self.order[0:i],col]) + np.sum(lower[self.order[i+1:],col])
            # Check if remainder is a valid probability
            diff = 1-probs
            if ((diff<=upper[self.order[i],col]) and (diff>=lower[self.order[i],col])):
              self.remainder = diff
              break
          # Assign final transition probabilities
          tempMat[self.order[0:i],col] = upper[self.order[0:i],col]
          tempMat[self.order[i+1:],col] = lower[self.order[i+1:],col]
          tempMat[self.order[i],col] = self.remainder
      # Construct final matrix for given action
      self.orderedTransitions.append(tempMat)

  # Perform a single iteration of algorithm
  def iterate(self):
    pTrans = np.zeros((len(self.states),len(self.actions)))
    # Do matrix-vector multiplication for each action
    for act in range(len(self.actions)):
      pTrans[:,act] = np.squeeze(np.transpose(self.orderedTransitions[act]) @ self.psats)
    # Maximize PSat for each state
    i=0
    for state in self.states:
      # Find valid actions
      idx = []
      for act in state.getActions():
        idx.append(self.actions.index(act))
      val = np.amax(pTrans[i,idx],axis=0)
      state.setPSat(np.amax(pTrans[i,idx],axis=0))
      idx = np.nonzero(pTrans[i,idx]==val)[0]
      acts = []
      for j in idx:
        acts.append(state.getActions()[j])
      state.setOptAct(acts)
      i=i+1
    return self.psats

  # Repeat value iteration until condition is met
  def valueIterationMin(self):
    # Find all available actions
    self.findActions()
    print("Find Actions")
    # Construct upper and lower transition matrices
    self.constructTransitions()
    print("Construct Transitions")
    # Get initial PSats
    self.getPSats()
    print("Get PSats")
    i = 0
    #print("Iteration " + str(i))
    #print(self.psats)
    oldPsats = self.psats
    # Construct ordered transition matrix
    self.orderMinTransitions()
    print("Order Min Transitions")
    # Perform iteration
    self.iterate()
    self.getPSats()
    i=i+1
    # While probabilities are not steady state
    print("Begin Loop")
    while(np.amax(np.abs(oldPsats-self.psats))>0.05):
      #print("Iteration " + str(i))
      #print(self.psats)
      oldPsats = self.psats
      # Iterate
      self.orderMinTransitions()
      self.iterate()
      self.getPSats()
      i=i+1
      print(np.amax(np.abs(oldPsats-self.psats)))
    print("End Loop")
    # Once probabilities are set, get optimal actions for each state
    self.getOptActs()
    #print("Optimal Actions: ")
    #print(self.optActs)

  # Simplified value iteration using precomputed state ordering
  def valueIterationMinSimple(self):
    # Find all available actions
    self.findActions()
    print("Find Actions")
    # Construct upper and lower transition matrices
    self.constructTransitions()
    print("Construct Transitions")
    # Get initial PSats
    self.getPSats()
    print("Get PSats")
    i = 0
    #print("Iteration " + str(i))
    #print(self.psats)
    oldPsats = self.psats
    # Construct ordered transition matrix
    self.orderMinTransitionsSimple(self.minOrder)
    print("Order Min Transitions")
    # Perform iteration
    self.iterate()
    self.getPSats()
    i=i+1
    # While probabilities are not steady state
    print("Begin Loop")
    while(np.amax(np.abs(oldPsats-self.psats))>0.05):
      #print("Iteration " + str(i))
      #print(self.psats)
      oldPsats = self.psats
      # Iterate
      #self.orderMinTransitions()
      self.iterate()
      self.getPSats()
      i=i+1
      print(np.amax(np.abs(oldPsats-self.psats)))
    print("End Loop")
    # Once probabilities are set, get optimal actions for each state
    self.getOptActs()
    #print("Optimal Actions: ")
    #print(self.optActs)

  # Repeat value iteration until condition is met
  def valueIterationMax(self):
    # Find all available actions
    self.findActions()
    # Construct upper and lower transition matrices
    self.constructTransitions()
    # Get initial PSats
    self.getPSats()
    i = 0
    #print("Iteration " + str(i))
    #print(self.psats)
    oldPsats = self.psats
    # Construct ordered transition matrix
    self.orderMaxTransitions()
    # Perform iteration
    self.iterate()
    self.getPSats()
    i=i+1
    # While probabilities are not steady state
    while(np.amax(np.abs(oldPsats-self.psats))>0.05):
      #print("Iteration " + str(i))
      #print(self.psats)
      oldPsats = self.psats
      # Iterate
      self.orderMaxTransitions()
      self.iterate()
      self.getPSats()
      i=i+1
    # Once probabilities are set, get optimal actions for each state
    self.getOptActs()
    #print("Optimal Actions: ")
    #print(self.optActs)
  
  # Simplified value iteration using precomputed state ordering
  def valueIterationMaxSimple(self):
    # Find all available actions
    self.findActions()
    # Construct upper and lower transition matrices
    self.constructTransitions()
    # Get initial PSats
    self.getPSats()
    i = 0
    #print("Iteration " + str(i))
    #print(self.psats)
    oldPsats = self.psats
    # Construct ordered transition matrix
    self.orderMaxTransitionsSimple(self.maxOrder)
    # Perform iteration
    self.iterate()
    self.getPSats()
    i=i+1
    # While probabilities are not steady state
    while(np.amax(np.abs(oldPsats-self.psats))>0.01):
      #print("Iteration " + str(i))
      #print(self.psats)
      oldPsats = self.psats
      # Iterate
      #self.orderMaxTransitions()
      self.iterate()
      self.getPSats()
      i=i+1
    # Once probabilities are set, get optimal actions for each state
    self.getOptActs()
    #print("Optimal Actions: ")
    #print(self.optActs)

  # Print states
  def getInfo(self):
    print("State Space\n")
    for state in self.states:
      state.getInfo()


# %% [markdown]
# True GP Terrain

# %%
# Gaussian Process Setup for True Terrain Dynamics
import math

class GaussianProcessTrueTerrain:

  # Constructor Method
  def __init__(self):
    # Initialize variance such that all known g(x) are within one standard deviation
    self.variance = 0.025**2
    # Initialize length
    self.l = 0.1
    # Initialize Data Points
    self.pos = np.empty((2,0))
    self.val = np.empty((1,0))
    self.ZZ = []
  
  # Add Data Samples
  def append(self,pos,val):
    # Append coordinate position
    self.pos = pos
    # Append g(x) value
    self.val = val

  # Construct Correlation Matrix
  def correlation(self, X1, X2):
    # Initialize correlation matrix
    K = np.zeros((X1.shape[1], X2.shape[1]))
    # Loop through all matrix entries
    for i in np.arange(X1.shape[1]):
      for j in np.arange(X2.shape[1]):
        K[i,j] = self.variance*math.exp(-(np.linalg.norm(X1[:,i]-X2[:,j])**2)/(2*self.l**2))
    return K
  
  # Calculate correlations
  def train(self):
    self.ZZ = np.linalg.inv(self.correlation(self.pos,self.pos))

  # Calculate predicted mean at a point
  def calcMean(self,pos):
    # edge case with no samples
    if self.pos.shape[1] == 0:
      return np.array([[0]])
    return self.correlation(pos,self.pos)@self.ZZ@self.val.T

  # Calculate predicted variance at a point
  def calcVariance(self, pos):
    # edge case with no samples
    if self.pos.shape[1] == 0:
      return np.array([[self.variance]])
    return self.correlation(pos,pos)-self.correlation(pos,self.pos)@self.ZZ@self.correlation(self.pos,pos)


# %%

# %%
# Sparse Gaussian Process
# Code adapted from Martin Krasser
import jax.numpy as jnp
import jax.scipy as jsp

from jax import random, jit, value_and_grad
from jax.config import config
from scipy.optimize import minimize

config.update("jax_enable_x64", True)

# SGP for Terrain Setup
class SparseGaussianProcessTerrain:

    # Constructor Method
  def __init__(self):
    # Initialize variance such that all known g(x) are within one standard deviation
    self.variance = 0.025 ** 2
    # Initialize length
    self.l = 0.1
    # Initialize Data Points
    self.pos = np.empty((2,0))
    self.val = np.empty((1,0))
    # Initialize stochastic noise variance
    self.noiseVariance = 0.01 ** 2
    self.noiseStd = 0.01
    self.ZZ = []
    # Initialize number of inducing variables
    self.m = 25
    self.theta_opt = []
  
  # Add Data Samples
  def append(self,pos,val):
    # Append coordinate position
    self.pos = pos
    # Append g(x) value
    self.val = val

  def calcMean(self,pos):
    # edge case with no samples
    if self.pos.shape[1] == 0:
      return np.array([[0]])
    f_test, f_test_cov = q_T(pos.T, self.theta_opt, self.X_m_opt, self.mu_m_opt, self.A_m_opt, self.K_mm_inv)
    return f_test

  def calcVariance(self,pos):
    # edge case with no samples
    if self.pos.shape[1] == 0:
      return np.array([[self.variance]])
    f_test, f_test_cov = q_T(pos.T, self.theta_opt, self.X_m_opt, self.mu_m_opt, self.A_m_opt, self.K_mm_inv)
    f_test_cov = np.where(f_test_cov<0,0,f_test_cov)
    f_test_var = np.diag(f_test_cov)
    f_test_std = np.sqrt(f_test_var) 
    return f_test_var

  def correlation(self, X1, X2):
    # Initialize correlation matrix
    K = np.zeros((X1.shape[1], X2.shape[1]))
    # Loop through all matrix entries
    for i in np.arange(X1.shape[1]):
      for j in np.arange(X2.shape[1]):
        K[i,j] = self.variance*math.exp(-(np.linalg.norm(X1[:,i]-X2[:,j])**2)/(2*self.l**2))
    return K
  
  def train(self):
    # Optimized kernel parameters and inducing inputs
    #self.theta_opt, self.X_m_opt = opt(self.pos,self.val,self.m,self.noiseStd)
    self.X_m_opt = opt_T(self.pos,self.val,self.m,self.noiseStd)
    self.mu_m_opt, self.A_m_opt, self.K_mm_inv = phi_opt_T(self.theta_opt, self.X_m_opt, self.pos.T, self.val.T, self.noiseStd)


# %% [markdown]
# 
# Sparse GP Terrain
# 

# %%

# %%
# Sparse Gaussian Process Methods for Terrain

# Kernel hyperparameters (length,standard deviation)
theta_fixed_T = jnp.array([0.1,0.25])
def kernel_T(X1, X2, theta):
    """
    Isotropic squared exponential kernel.
    
    Args:
        X1: Array of m points (m, d).
        X2: Array of n points (n, d).
        theta: kernel parameters (2,)
    """
    sqdist = jnp.sum(X1 ** 2, 1).reshape(-1,1) + jnp.sum(X2 ** 2, 1) - 2 * jnp.dot(X1, X2.T)
    return theta[1] ** 2 * jnp.exp(-0.5 / theta[0] ** 2 * sqdist)


def kernel_diag_T(d, theta):
    """
    Isotropic squared exponential kernel (computes diagonal elements only).
    """
    return jnp.full(shape=d, fill_value=theta[0] ** 2)

def jitter(d, value=1e-6):
    return jnp.eye(d) * value


def softplus(X):
    return jnp.log(1 + jnp.exp(X))


def softplus_inv(X):
    return jnp.log(jnp.exp(X) - 1)


def pack_params(theta, X_m):
    return jnp.concatenate([softplus_inv(theta), X_m.ravel()])


def unpack_params_T(params):
    return softplus(params[:2]), jnp.array(params[2:].reshape(-1, 2))


def nlb_fn_T(X, y, sigma_y):
    n = X.shape[0]

    def nlb(params):
        """
        Negative lower bound on log marginal likelihood.
        
        Args:
            params: kernel parameters `theta` and inducing inputs `X_m`
        """
        
        theta, X_m = unpack_params_T(params)
        K_mm = kernel_T(X_m, X_m, theta) + jitter(X_m.shape[0])
        K_mn = kernel_T(X_m, X, theta)

        L = jnp.linalg.cholesky(K_mm)  # m x m
        A = jsp.linalg.solve_triangular(L, K_mn, lower=True) / sigma_y # m x n        
        AAT = A @ A.T  # m x m
        B = jnp.eye(X_m.shape[0]) + AAT  # m x m
        LB = jnp.linalg.cholesky(B)  # m x m
        c = jsp.linalg.solve_triangular(LB, A.dot(y), lower=True) / sigma_y  # m x 1

        # Equation (13)
        lb = - n / 2 * jnp.log(2 * jnp.pi)
        lb -= jnp.sum(jnp.log(jnp.diag(LB)))
        lb -= n / 2 * jnp.log(sigma_y ** 2)
        lb -= 0.5 / sigma_y ** 2 * y.T.dot(y)
        lb += 0.5 * c.T.dot(c)
        lb -= 0.5 / sigma_y ** 2 * jnp.sum(kernel_diag_T(n, theta))
        lb += 0.5 * jnp.trace(AAT)

        return -lb[0, 0]

    # nlb_grad returns the negative lower bound and 
    # its gradient w.r.t. params i.e. theta and X_m.
    nlb_grad = jit(value_and_grad(nlb))

    def nlb_grad_wrapper(params):
        value, grads = nlb_grad(params)
        # scipy.optimize.minimize cannot handle
        # JAX DeviceArray directly. a conversion
        # to Numpy ndarray is needed.
        return np.array(value), np.array(grads)

    return nlb_grad_wrapper

  # Run optimization
def opt_T(pos,val,m,noiseStd):
  # Initialize inducing inputs
  indices = jnp.floor(jnp.linspace(0,pos.shape[1]-1,m)).astype(int)
  X_m = jnp.array(pos.T[indices,:])
  res = minimize(fun=nlb_fn_T(pos.T, val.T, noiseStd),
                  x0=pack_params(theta_fixed_T, X_m),
                  method='L-BFGS-B',
                  jac=True)

    # Optimized kernel parameters and inducing inputs
  theta_opt, X_m_opt = unpack_params_T(res.x)
  return X_m_opt  

@jit
def phi_opt_T(theta, X_m, X, y, sigma_y):
  theta = theta_fixed_T
  """Optimize mu_m and A_m using Equations (11) and (12)."""
  precision = (1.0 / sigma_y ** 2)

  K_mm = kernel_T(X_m, X_m, theta) + jitter(X_m.shape[0])
  K_mm_inv = jnp.linalg.inv(K_mm)
  K_nm = kernel_T(X, X_m, theta)
  K_mn = K_nm.T
    
  Sigma = jnp.linalg.inv(K_mm + precision * K_mn @ K_nm)
    
  mu_m = precision * (K_mm @ Sigma @ K_mn).dot(y)
  A_m = K_mm @ Sigma @ K_mm    
  
  return mu_m, A_m, K_mm_inv


@jit
def q_T(X_test, theta, X_m, mu_m, A_m, K_mm_inv):
  """
  Approximate posterior. 
    
  Computes mean and covariance of latent 
  function values at test inputs X_test.
  """
  theta = theta_fixed_T
  K_ss = kernel_T(X_test, X_test, theta)
  K_sm = kernel_T(X_test, X_m, theta)
  K_ms = K_sm.T

  f_q = (K_sm @ K_mm_inv).dot(mu_m)
  f_q_cov = K_ss - K_sm @ K_mm_inv @ K_ms + K_sm @ K_mm_inv @ A_m @ K_mm_inv @ K_ms
    
  return f_q, f_q_cov






# %%



# %%
# Sparse Gaussian Process for Model Error
# Code adapted from Martin Krasser
noiseStd = 2.5
# Compensate for yaw error with nonzero mean
meanYawError = 7.5

class SparseGaussianProcessModel:

    # Constructor Method
  def __init__(self):
    # Initialize variance such that all known g(x) are within one standard deviation
    self.variance = meanYawError ** 2
    # Initialize length
    self.l = [1,5,0.1,0.5]
    # Initialize Data Points
    self.pos = np.empty((3,0))
    self.val = np.empty((1,0))
    # Initialize stochastic noise variance
    self.noiseVariance = noiseStd ** 2
    self.noiseStd = noiseStd
    self.ZZ = []
    # Initialize number of inducing variables
    self.m = 5
    self.theta_opt = []
  
  # Add Data Samples
  def append(self,pos,val):
    # Append coordinate position
    self.pos = pos
    # Append g(x) value
    self.val = val

  def calcMean(self,pos):
    # edge case with no samples
    if self.pos.shape[1] == 0:
      return np.array([[0]])
    f_test, f_test_cov = q_M(pos.T, self.theta_opt, self.X_m_opt, self.mu_m_opt, self.A_m_opt, self.K_mm_inv)
    if math.isnan(f_test):
      return 0
    if f_test>meanYawError:
      f_test=meanYawError
    elif f_test <-meanYawError:
      f_test=-meanYawError
    return f_test

  def calcVariance(self,pos):
    # edge case with no samples
    if self.pos.shape[1] == 0:
      return np.array([[self.variance]])
    f_test, f_test_cov = q_M(pos.T, self.theta_opt, self.X_m_opt, self.mu_m_opt, self.A_m_opt, self.K_mm_inv)
    f_test_cov = np.where(f_test_cov<0,0,f_test_cov)
    f_test_var = np.diag(f_test_cov)
    f_test_std = np.sqrt(f_test_var)
    #print(f_test_var)
    return f_test_std

  def correlation(self, X1, X2):
    # Initialize correlation matrix
    # K = np.zeros((X1.shape[1], X2.shape[1]))
    # # Loop through all matrix entries
    # for i in np.arange(X1.shape[1]):
    #   for j in np.arange(X2.shape[1]):
    #     diff = np.mod(X1[:,i]-X2[:,j],180)
    #     if diff[1]>180:
    #       diff[1]=360-diff[1]
    #     K[i,j] = self.variance*math.exp(-0.5*(diff@make_diag(self.l)@diff))
    theta_fixed = jnp.array([0.2,15,0.1,5])
    return np.array(kernel_M(X1.T,X2.T,theta_fixed))
  
  def train(self):
    # Optimized kernel parameters and inducing inputs
    #self.theta_opt, self.X_m_opt = opt(self.pos,self.val,self.m,self.noiseStd)
    self.X_m_opt = opt_M(self.pos,self.val,self.m,self.noiseStd)
    self.mu_m_opt, self.A_m_opt, self.K_mm_inv = phi_opt_M(self.theta_opt, self.X_m_opt, self.pos.T, self.val.T, self.noiseStd)


# %% [markdown]
# 
# Sparse GP Model Error
# 

# %%

# Sparse Gaussian Process Methods for Model Error GP

# Kernel hyperparameters (length,standard deviation)
# d, theta, z, foot
theta_fixed_M = jnp.array([1,15,0.1,0.5,meanYawError*1.25])
def kernel_M(X1, X2, theta):
    """
    Anisotropic squared exponential kernel.
    
    Args:
        X1: Array of m points (m, d).
        X2: Array of n points (n, d).
        theta: kernel parameters (5,)
    """
    sqdist0 = ((X1[:,0] ** 2).reshape(-1,1) + (X2[:,0] ** 2).reshape(1,-1) - 2 * X1[:,0].reshape(-1,1)@X2[:,0].reshape(1,-1))/(theta[0]**2)
    sqdist1 = ((X1[:,1] ** 2).reshape(-1,1) + (X2[:,1] ** 2).reshape(1,-1) - 2 * X1[:,1].reshape(-1,1)@X2[:,1].reshape(1,-1))
    sqdist1 = jnp.where(sqdist1>180**2,(360-jnp.sqrt(sqdist1))**2,sqdist1)
    sqdist1 = sqdist1/(theta[1]**2)
    sqdist2 = ((X1[:,2] ** 2).reshape(-1,1) + (X2[:,2] ** 2).reshape(1,-1) - 2 * X1[:,2].reshape(-1,1)@X2[:,2].reshape(1,-1))/(theta[2]**2)
    sqdist3 = ((X1[:,3] ** 2).reshape(-1,1) + (X2[:,3] ** 2).reshape(1,-1) - 2 * X1[:,3].reshape(-1,1)@X2[:,3].reshape(1,-1))/(theta[3]**2)
    return theta[1] ** 2 * jnp.exp(-0.5 * (sqdist0+sqdist1+sqdist2+sqdist3))

def kernel_diag_M(d, theta):
    """
    Isotropic squared exponential kernel (computes diagonal elements only).
    """
    return jnp.full(shape=d, fill_value=theta[0:-1] ** 2)

# Create diagonal matrix of length parameters
def make_diag(theta):
    return jnp.diag(1.0/(theta[0:-1]**2))

def unpack_params_M(params):
    return softplus(params[:5]), jnp.array(params[5:].reshape(-1, 4))

def nlb_fn_M(X, y, sigma_y):
    n = X.shape[1]

    def nlb(params):
        """
        Negative lower bound on log marginal likelihood.
        
        Args:
            params: kernel parameters `theta` and inducing inputs `X_m`
        """
        
        theta, X_m = unpack_params_M(params)
        K_mm = kernel_M(X_m, X_m, theta) + jitter(X_m.shape[0])
        K_mn = kernel_M(X_m, X, theta)

        L = jnp.linalg.cholesky(K_mm)  # m x m
        A = jsp.linalg.solve_triangular(L, K_mn, lower=True) / sigma_y # m x n        
        AAT = A @ A.T  # m x m
        B = jnp.eye(X_m.shape[0]) + AAT  # m x m
        LB = jnp.linalg.cholesky(B)  # m x m
        c = jsp.linalg.solve_triangular(LB, A.dot(y), lower=True) / sigma_y  # m x 1

        # Equation (13)
        lb = - n / 2 * jnp.log(2 * jnp.pi)
        lb -= jnp.sum(jnp.log(jnp.diag(LB)))
        lb -= n / 2 * jnp.log(sigma_y ** 2)
        lb -= 0.5 / sigma_y ** 2 * y.T.dot(y)
        lb += 0.5 * c.T.dot(c)
        lb -= 0.5 / sigma_y ** 2 * jnp.sum(kernel_diag_M(n, theta))
        lb += 0.5 * jnp.trace(AAT)

        return -lb[0, 0]

    # nlb_grad returns the negative lower bound and 
    # its gradient w.r.t. params i.e. theta and X_m.
    nlb_grad = jit(value_and_grad(nlb))

    def nlb_grad_wrapper(params):
        value, grads = nlb_grad(params)
        # scipy.optimize.minimize cannot handle
        # JAX DeviceArray directly. a conversion
        # to Numpy ndarray is needed.
        return np.array(value), np.array(grads)

    return nlb_grad_wrapper

  # Run optimization
def opt_M(pos,val,m,noiseStd):
  # Initialize inducing inputs
  indices = jnp.floor(jnp.linspace(0,pos.shape[1]-1,m)).astype(int)
  X_m = jnp.array(pos.T[indices,:])
  res = minimize(fun=nlb_fn_M(pos.T, val.T, noiseStd),
                  x0=pack_params(jnp.array(theta_fixed_M), X_m),
                  method='L-BFGS-B',
                  jac=True)

    # Optimized kernel parameters and inducing inputs
  theta_opt, X_m_opt = unpack_params_M(res.x)
  return X_m_opt  

  # Run optimization
@jit
def phi_opt_M(theta, X_m, X, y, sigma_y):
  theta = theta_fixed_M
  """Optimize mu_m and A_m using Equations (11) and (12)."""
  precision = (1.0 / sigma_y ** 2)

  K_mm = kernel_M(X_m, X_m, theta) + jitter(X_m.shape[0])
  K_mm_inv = jnp.linalg.inv(K_mm)
  K_nm = kernel_M(X, X_m, theta)
  K_mn = K_nm.T
    
  Sigma = jnp.linalg.inv(K_mm + precision * K_mn @ K_nm)
    
  mu_m = precision * (K_mm @ Sigma @ K_mn).dot(y)
  A_m = K_mm @ Sigma @ K_mm    
  
  return mu_m, A_m, K_mm_inv


@jit
def q_M(X_test, theta, X_m, mu_m, A_m, K_mm_inv):
  """
  Approximate posterior. 
    
  Computes mean and covariance of latent 
  function values at test inputs X_test.
  """
  theta = theta_fixed_M
  K_ss = kernel_M(X_test, X_test, theta)
  K_sm = kernel_M(X_test, X_m, theta)
  K_ms = K_sm.T

  f_q = (K_sm @ K_mm_inv).dot(mu_m)
  f_q_cov = K_ss - K_sm @ K_mm_inv @ K_ms + K_sm @ K_mm_inv @ A_m @ K_mm_inv @ K_ms
    
  return f_q, f_q_cov


# %%
# %%
# Sparse Gaussian Process for Yaw Error
# Code adapted from Martin Krasser
padding = 1
class SparseGaussianProcessYaw:

    # Constructor Method
  def __init__(self):
    # Initialize variance such that all known g(x) are within one standard deviation
    self.variance = (meanYawError+noiseStd+padding) ** 2
    # Initialize length
    self.l = [0.1,2.5]
    # Initialize Data Points
    self.pos = np.empty((2,0))
    self.val = np.empty((1,0))
    # Initialize stochastic noise variance
    self.noiseVariance = noiseStd ** 2
    self.noiseStd = noiseStd
    self.ZZ = []
    # Initialize number of inducing variables
    self.m = 5
    self.theta_opt = []
  
  # Add Data Samples
  def append(self,pos,val):
    # Append coordinate position
    self.pos = pos
    # Append g(x) value
    self.val = abs(val)# - meanYawError

  def calcMean(self,pos):
    # edge case with no samples
    if self.pos.shape[1] == 0:
      return np.array([[0]]) + meanYawError + noiseStd + padding
    f_test, f_test_cov = q_Y(pos.T, self.theta_opt, self.X_m_opt, self.mu_m_opt, self.A_m_opt, self.K_mm_inv)
    #print(f_test+meanYawError)
    if math.isnan(f_test):
      print(pos)
    # if f_test < -meanYawError - noiseStd:
    #   f_test = -meanYawError-noiseStd
    if f_test > meanYawError + noiseStd + padding:
      f_test = meanYawError + noiseStd
    if f_test < 0:
      f_test = 0
    return f_test

  def calcVariance(self,pos):
    # edge case with no samples
    if self.pos.shape[1] == 0:
      return np.array([[self.variance]])
    f_test, f_test_cov = q_Y(pos.T, self.theta_opt, self.X_m_opt, self.mu_m_opt, self.A_m_opt, self.K_mm_inv)
    f_test_cov = np.where(f_test_cov<0,0,f_test_cov)
    f_test_var = np.diag(f_test_cov)
    f_test_std = np.sqrt(f_test_var) 
    print(f_test_var)
    return f_test_var

  def correlation(self, X1, X2):
    # Initialize correlation matrix
    # K = np.zeros((X1.shape[1], X2.shape[1]))
    # # Loop through all matrix entries
    # for i in np.arange(X1.shape[1]):
    #   for j in np.arange(X2.shape[1]):
    #     K[i,j] = self.variance*math.exp(-0.5*((X1[:,i]-X2[:,j])@make_diag(self.l)@(X1[:,i]-X2[:,j])))
    theta_fixed = jnp.array([0.1,5,5])
    return np.array(kernel_Y(X1.T,X2.T,theta_fixed))
  
  def train(self):
    # Optimized kernel parameters and inducing inputs
    #self.theta_opt, self.X_m_opt = opt(self.pos,self.val,self.m,self.noiseStd)
    self.X_m_opt = opt_Y(self.pos,self.val,self.m,self.noiseStd)
    self.mu_m_opt, self.A_m_opt, self.K_mm_inv = phi_opt_Y(self.theta_opt, self.X_m_opt, self.pos.T, self.val.T, self.noiseStd)



# %% [markdown]
# 
# Sparse GP Yaw Error
# 

# %%

# %%
# Sparse Gaussian Process Methods for Yaw Error GP

# Kernel hyperparameters (length,standard deviation)
theta_fixed_Y = jnp.array([2.5,90,meanYawError+noiseStd+padding])
def kernel_Y(X1, X2, theta):
    """
    Anisotropic squared exponential kernel.
    
    Args:
        X1: Array of m points (m, d).
        X2: Array of n points (n, d).
        theta: kernel parameters (3,)
    """
    sqdist0 = ((X1[:,0] ** 2).reshape(-1,1) + (X2[:,0] ** 2).reshape(1,-1) - 2 * X1[:,0].reshape(-1,1)@X2[:,0].reshape(1,-1))/(theta[0]**2)
    sqdist1 = ((X1[:,1] ** 2).reshape(-1,1) + (X2[:,1] ** 2).reshape(1,-1)- 2 * X1[:,1].reshape(-1,1)@X2[:,1].reshape(1,-1))/(theta[1]**2)
    return theta[1] ** 2 * jnp.exp(-0.5 * (sqdist0+sqdist1))

def kernel_diag_Y(d, theta):
    """
    Isotropic squared exponential kernel (computes diagonal elements only).
    """
    return jnp.full(shape=d, fill_value=theta[0:-1] ** 2)

def unpack_params_Y(params):
    return softplus(params[:3]), jnp.array(params[3:].reshape(-1, 2))

def nlb_fn_Y(X, y, sigma_y):
    n = X.shape[1]

    def nlb(params):
        """
        Negative lower bound on log marginal likelihood.
        
        Args:
            params: kernel parameters `theta` and inducing inputs `X_m`
        """
        
        theta, X_m = unpack_params_Y(params)
        K_mm = kernel_Y(X_m, X_m, theta) + jitter(X_m.shape[0])
        K_mn = kernel_Y(X_m, X, theta)

        L = jnp.linalg.cholesky(K_mm)  # m x m
        A = jsp.linalg.solve_triangular(L, K_mn, lower=True) / sigma_y # m x n        
        AAT = A @ A.T  # m x m
        B = jnp.eye(X_m.shape[0]) + AAT  # m x m
        LB = jnp.linalg.cholesky(B)  # m x m
        c = jsp.linalg.solve_triangular(LB, A.dot(y), lower=True) / sigma_y  # m x 1

        # Equation (13)
        lb = - n / 2 * jnp.log(2 * jnp.pi)
        lb -= jnp.sum(jnp.log(jnp.diag(LB)))
        lb -= n / 2 * jnp.log(sigma_y ** 2)
        lb -= 0.5 / sigma_y ** 2 * y.T.dot(y)
        lb += 0.5 * c.T.dot(c)
        lb -= 0.5 / sigma_y ** 2 * jnp.sum(kernel_diag_Y(n, theta))
        lb += 0.5 * jnp.trace(AAT)

        return -lb[0, 0]

    # nlb_grad returns the negative lower bound and 
    # its gradient w.r.t. params i.e. theta and X_m.
    nlb_grad = jit(value_and_grad(nlb))

    def nlb_grad_wrapper(params):
        value, grads = nlb_grad(params)
        # scipy.optimize.minimize cannot handle
        # JAX DeviceArray directly. a conversion
        # to Numpy ndarray is needed.
        return np.array(value), np.array(grads)

    return nlb_grad_wrapper

  # Run optimization
def opt_Y(pos,val,m,noiseStd):
  # Initialize inducing inputs
  indices = jnp.floor(jnp.linspace(0,pos.shape[1]-1,m)).astype(int)
  X_m = jnp.array(pos.T[indices,:])
  res = minimize(fun=nlb_fn_Y(pos.T, val.T, noiseStd),
                  x0=pack_params(jnp.array(theta_fixed_Y), X_m),
                  method='L-BFGS-B',
                  jac=True)

    # Optimized kernel parameters and inducing inputs
  theta_opt, X_m_opt = unpack_params_Y(res.x)
  return X_m_opt  

  # Run optimization
@jit
def phi_opt_Y(theta, X_m, X, y, sigma_y):
  theta = theta_fixed_Y
  """Optimize mu_m and A_m using Equations (11) and (12)."""
  precision = (1.0 / sigma_y ** 2)

  K_mm = kernel_Y(X_m, X_m, theta) + jitter(X_m.shape[0])
  K_mm_inv = jnp.linalg.inv(K_mm)
  K_nm = kernel_Y(X, X_m, theta)
  K_mn = K_nm.T
    
  Sigma = jnp.linalg.inv(K_mm + precision * K_mn @ K_nm)
    
  mu_m = precision * (K_mm @ Sigma @ K_mn).dot(y)
  A_m = K_mm @ Sigma @ K_mm    
  
  return mu_m, A_m, K_mm_inv


@jit
def q_Y(X_test, theta, X_m, mu_m, A_m, K_mm_inv):
  """
  Approximate posterior. 
    
  Computes mean and covariance of latent 
  function values at test inputs X_test.
  """
  theta = theta_fixed_Y
  K_ss = kernel_Y(X_test, X_test, theta)
  K_sm = kernel_Y(X_test, X_m, theta)
  K_ms = K_sm.T

  f_q = (K_sm @ K_mm_inv).dot(mu_m)
  f_q_cov = K_ss - K_sm @ K_mm_inv @ K_ms + K_sm @ K_mm_inv @ A_m @ K_mm_inv @ K_ms
    
  return f_q, f_q_cov


# %% [markdown]
# 
# Case Study Setup
# 

# %%

# %%
# Multi Step Goal Example Setup

import scipy.stats as stats
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.animation as animation
import copy

# Initialize random seed for rng
rng = np.random.default_rng(2022)


#Initialize Plotting
# img = plt.imread("legged_robot_multistep_casestudy.png")
# f3 = plt.figure()
# plt.imshow(img,extent=[0,2,0,2])
# plt.xticks(np.arange(0, gridSize*stepLength+stepLength, step=stepLength))
# plt.yticks(np.arange(0, gridSize*stepLength+stepLength, step=stepLength))
# f3.tight_layout()

com_xs = []
com_ys = []
foot_xs = []
foot_ys = []
frame_idxs = []
traj_xs = []
traj_ys = []
frame_count = 0
compression = 20
# prev_com = plt.scatter([], [], s=10, c="black",alpha=0.6, marker='o')
# curr_com = plt.scatter([], [], s=10, c="blue", alpha=0.6, marker = 'o')
# prev_feet = plt.scatter([], [], s=10, c="indianred", alpha = 0.6, marker = 'o')
# curr_feet = plt.scatter([], [], s=10, c="darkred", alpha = 0.6, marker = 'o')
# com_traj, = plt.plot([], [], c="indigo", lw=1)

frame_curr_idx = 0
# Function to create animated video
# def animate(i):
#   i = i*compression
#   if i>frame_count:
#     i=frame_count
#   global frame_curr_idx
#   # Plot continuous CoM trajectory
#   com_traj.set_data(traj_xs[0:i],traj_ys[0:i])
#   # Plot CoM/Foot Apex
#   if i == 0:
#     curr_com.set_offsets(np.c_[com_xs[0],com_ys[0]])
#     curr_feet.set_offsets(np.c_[foot_xs[0],foot_ys[0]])
#   else:
#     if frame_curr_idx < len(frame_idxs)-1 and i>=frame_idxs[frame_curr_idx+1]:
#       curr_feet.set_offsets(np.c_[foot_xs[frame_curr_idx],foot_ys[frame_curr_idx]])
#       prev_feet.set_offsets(np.c_[foot_xs[0:frame_curr_idx],foot_ys[0:frame_curr_idx]])
#       prev_com.set_offsets(np.c_[com_xs[0:frame_curr_idx],com_ys[0:frame_curr_idx]])
#       curr_com.set_offsets(np.c_[com_xs[frame_curr_idx],com_ys[frame_curr_idx]])
#       frame_curr_idx = frame_curr_idx + 1
#   return prev_com, curr_com, prev_feet, curr_feet, com_traj

# Create sample goal class
class MultiStepGoal:
  # Create random terrain elevations (between -0.1 and 0.1)
  terrainE = rng.integers(low=-1,high=2,size=(1,gridSize**2)).astype(float)/10

  # Constructor Method
  def __init__(self):
    # Number of states
    self.numStates = numStates
    # Initialize position of robot (x,y,yaw)
    self.currentPos = np.array([[stateLength/2],[stateLength/2],[angleSize/2],[0]])
    self.waypoint = self.currentPos
    ang = self.currentPos[2]*math.pi/180-math.pi/2*(1-self.currentPos[3])+math.pi/2*self.currentPos[3]
    dxdy = self.currentPos[0:2]+stateLength/2*np.array([[math.cos(ang)],[math.sin(ang)]])
    self.waypoint[0]=dxdy[0]
    self.waypoint[1]=dxdy[1]
    self.vapex = 0.35
    self.zapex = 0.983
    self.time = 0
    # Calculate region centers
    xpos = np.reshape(np.arange(stateLength/2,gridSize*stateLength,stateLength),(1,gridSize))
    ypos = np.ones((1,gridSize))
    self.centers = np.concatenate((np.concatenate((xpos,xpos,xpos,xpos,xpos,xpos,xpos,xpos,xpos,xpos),axis=1),np.reshape(xpos.T@ypos,(1,gridSize ** 2))),axis=0)
    # Initialize all Gaussian Processes
    self.gpT = SparseGaussianProcessTerrain()
    self.gpM = SparseGaussianProcessModel()
    self.gpY = SparseGaussianProcessYaw()
    # Initialize Gaussian Processes for true dynamics
    self.gT = GaussianProcessTrueTerrain()
    self.gT.append(self.centers, MultiStepGoal.terrainE)
    self.gT.train()
    # Find current foot position
    self.footPos = np.array([[stateLength/2],[stateLength/2],[self.getElevationA(np.array([[stateLength/2],[stateLength/2]])).item()]])
    self.prevFoot = self.footPos
    self.prevYaw = self.currentPos[2]
    # Initialize stochastic noise
    self.noiseStd = noiseStd
    self.noiseVariance = self.noiseStd ** 2
    self.v = stats.truncnorm(-30,30,loc=0,scale=self.noiseStd)
    # Initialize GP training arrays
    self.Tpos = np.empty((2,0))
    self.Tval = np.empty((1,0))
    self.Mpos = np.empty((4,0))
    self.Mval = np.empty((1,0))
    self.Ypos = np.empty((2,0))
    self.Yval = np.empty((1,0))
    # Set initial state
    self.initState = self.getStateID(self.currentPos)
    self.currState = self.initState
    # Set desired probability of satisfaction for goal state
    self.Psat = 1
    # Initialize uncertainty vector
    self.uncertainty = np.empty((1,0))
    # Initialize vector of psats
    self.psats = np.empty((1,0))
    # Initialize error vector
    self.errors = np.empty((1,0))
    # Initialize iteration tracker
    self.stepCounter = 0
    self.stepIter = [0]

  # Method to determine current abstraction xy region
  def getRegion(self,pos):
    pos = np.where(pos==stateLength*gridSize,pos-0.01,pos)
    # Get x value
    x = np.floor(pos[0]/stateLength)
    # Get y value
    y = np.floor(pos[1]/stateLength)
    # Round down if position is exactly 1
    x = np.min(np.concatenate((x.reshape(1,1),np.array([[int(gridSize)]]))))
    y = np.min(np.concatenate((y.reshape(1,1),np.array([[int(gridSize)]]))))
    # Return region
    return max(int(gridSize*y+x),-1)

  # Method to determine current heading bin
  def getHeadingBin(self):
    return np.floor(self.currentPos[2]/angleSize)

  # Method to determine current IMDP State
  def getStateID(self,pos):
    if (pos[0] < 0) or (pos[0] > stateLength*gridSize) or (pos[1] < 0) or (pos[1] > stateLength*gridSize):
      return -1
    return int(self.getRegion(pos[0:2])*angleLength*2+math.floor(pos[2]/angleSize)*2+pos[3])

  # Method to calculate true unknown terrain elevation at arbitrary state
  def getElevationA(self, pos):
    # Get true GP values
    t=self.gT.calcMean(pos)
    return np.array(min(max(t,-.1),.1)).reshape(1,1)
        
  # Method to set the current position
  def setPos(self,pos):
    self.currentPos=pos

  # Method to return the current position
  def getPos(self):
    return self.currentPos.reshape(3,1)

  # Method to return center of region
  def getCenter(self,state):
    id = state.getID() % numStates
    xy = self.centers[:,int(np.floor(id/(angleLength*2)))].reshape(2,1)
    angle = np.floor(np.mod(id,angleLength*2)/2) * angleSize + (angleSize/2.0)
    # 0 for left foot, 1 for right foot
    fs = np.mod(id,2)
    return np.concatenate((xy,np.array([[angle]]),np.array([[fs]])),axis=0).reshape(4,1)

  # Method to simulate a discrete step
  def step(self,state,action):
    # Calculate controller
    idx = state.actions.index(action)
    prevReg = self.getRegion(self.currentPos)
    tcen = self.getCenter(self.s.states[state.targets[idx]])
    print("Target Region")
    print(self.getRegion(tcen))
    #self.counter = self.s.states[state.targets[idx]].getCounter()
    # Adjust foot position for yaw error
    f_d = math.sqrt((self.currentPos[0]-self.footPos[0])**2 + (self.currentPos[1]-self.footPos[1])**2)
    ang = self.currentPos[2]*math.pi/180+math.pi/2*(1-self.currentPos[3])-math.pi/2*self.currentPos[3]
    dxdy = self.currentPos[0:2]+f_d*np.array([[math.cos(ang)],[math.sin(ang)]])
    self.footPos[0]=dxdy[0]
    self.footPos[1]=dxdy[1]
    # Temporary test neglecting foot elevation
    self.footPos[2] = 0
    #self.footPos[2]=self.getElevationA(self.footPos[0:2])
    self.dz = state.zave[idx]-self.footPos[2]
    self.dd = math.sqrt((tcen[0]-self.currentPos[0])**2 + (tcen[1]-self.currentPos[1])**2)
    self.dt = tcen[2]-self.currentPos[2]
    while self.dt > 180:
      self.dt = self.dt-360
    while self.dt < -180:
      self.dt = self.dt + 360
    if action == 'b':
      ang=self.currentPos[2]*math.pi/180
      self.currentPos[2] = tcen[2]
      self.waypoint = copy.deepcopy(self.currentPos)
      self.footPos[0]=self.footPos[0]+2*(self.currentPos[0]-self.footPos[0])
      self.footPos[1]=self.footPos[1]+2*(self.currentPos[1]-self.footPos[1])
      if self.footPos[0] > gridSize*stateLength:
        self.footPos[0] = gridSize*stateLength
      if self.footPos[1] > gridSize*stateLength:
        self.footPos[1] = gridSize*stateLength
      if self.footPos[0] < 0:
        self.footPos[0] = 0
      if self.footPos[1] < 0:
        self.footPos[1] = 0
      self.footPos[2]=self.getElevationA(self.footPos[0:2])
      self.prevFoot = self.footPos
      self.prevFoot[0]=self.footPos[0]+2*(self.currentPos[0]-self.footPos[0])
      self.prevFoot[1]=self.footPos[1]+2*(self.currentPos[1]-self.footPos[1])
      self.prevFoot[2]=self.getElevationA(self.prevFoot[0:2])
      self.vapex = 0.35
      angd = 0
      # Write foot placements to induce good visualization of turn-in-place case
      if self.dt >= 0:
        angd = np.linspace(0,self.dt,6)
      else:
        angd = np.linspace(0,self.dt,6)
      if self.currentPos[3]==0:
        lf = copy.deepcopy(self.footPos)
        rf = copy.deepcopy(self.footPos)
        prevA = 0
        for ad in angd:
          a = ad*math.pi/180
          rf[0:2] = self.currentPos[0:2]-footSpace*np.array([[math.cos(ang+prevA+math.pi/2)],[math.sin(ang+prevA+math.pi/2)]])
          # rf[2] = self.getElevationA(self.rf[0:2])
          #self.writeFeet(lf,rf,np.array([3]),self.currentPos[2],self.currentPos[2])
          self.writeFeet(rf,np.array([ang+prevA]))
          if rf[1] > gridSize*stateLength:
            rf[1] = gridSize*stateLength
          if rf[0] < 0:
            rf[0] = 0
          if self.footPos[1] < 0:
            rf[1] = 0
          lf[0:2] = self.currentPos[0:2]+footSpace*np.array([[math.cos(ang+a+math.pi/2)],[math.sin(ang+a+math.pi/2)]])
          # lf[2] = self.getElevationA(self.rf[0:2])
          #self.writeFeet(lf,rf,np.array([3]),self.currentPos[2],self.currentPos[2])
          self.writeFeet(lf,np.array([ang+a]))
          prevA = a
        rf[0:2] = self.currentPos[0:2]-footSpace*np.array([[math.cos(ang+self.dt)],[math.sin(ang+self.dt)]])
        # rf[2] = self.getElevationA(self.rf[0:2])
        #self.writeFeet(lf,rf,np.array([3]),self.currentPos[2],self.currentPos[2])
        #self.writeFeet(rf)
        self.prevFoot=copy.deepcopy(rf)
      else:
        rf = copy.deepcopy(self.footPos)
        lf = copy.deepcopy(self.footPos)
        prevA = 0
        for ad in angd:
          a = ad*math.pi/180
          lf[0:2] = self.currentPos[0:2]-footSpace*np.array([[math.cos(ang+prevA-math.pi/2)],[math.sin(ang+prevA-math.pi/2)]])
          # lf[2] = self.getElevationA(self.rf[0:2])
          #self.writeFeet(lf,rf,np.array([3]),self.currentPos[2],self.currentPos[2])
          self.writeFeet(lf,np.array([ang+prevA]))
          rf[0:2] = self.currentPos[0:2]+footSpace*np.array([[math.cos(ang+a-math.pi/2)],[math.sin(ang+a-math.pi/2)]])
          # rf[2] = self.getElevationA(self.rf[0:2])
          #self.writeFeet(lf,rf,np.array([3]),self.currentPos[2],self.currentPos[2])
          self.writeFeet(rf,np.array([ang+a]))
          prevA = a
        lf[0:2] = self.currentPos[0:2]-footSpace*np.array([[math.cos(ang+self.dt-math.pi/2)],[math.sin(ang+self.dt-math.pi/2)]])
        # lf[2] = self.getElevationA(self.rf[0:2])
        #self.writeFeet(lf,rf,np.array([3]),self.currentPos[2],self.currentPos[2])
        #self.writeFeet(lf)
        self.prevFoot=copy.deepcopy(lf)
      self.dt = np.array([0])
    targetY = self.dt+self.currentPos[2]
    ddt = 0
    #for k in range(20):
    #  ddt = -self.gpM.calcMean(np.array([[self.dd],[self.dt.item()+float(ddt)],[self.dz.item()],[self.currentPos[3].item()]]))
    self.dt = self.dt + ddt
    self.zUncertainty = state.zUncertainty[idx]
    # Adjust target waypoint to match desired heading angle
    self.desAng = self.dt+self.currentPos[2]
    prevPos = copy.deepcopy(self.currentPos)
    if action=='s':
      #self.prevFoot = copy.deepcopy(self.footPos)
      #self.prevYaw = copy.deepcopy(self.currentPos[2])
      self.angError=self.yawError(self.dd,self.dt,self.dz,self.footPos[2]-self.footPos[2],tcen[3])
      self.currentPos[2]=self.currentPos[2]+self.dt+self.angError
      self.currentPos[3]=1-self.currentPos[3]
      self.angError = np.array([self.angError])
      #if self.currentPos[3]==0:
        #self.writeFeet(self.footPos,self.prevFoot,np.array([1.5]),self.currentPos[2],self.prevYaw)
      #  self.writeFeet(self.footPos,self.prevFoot,np.array([3]),self.currentPos[2],self.currentPos[2])
      #else:
        #self.writeFeet(self.prevFoot,self.footPos,np.array([1.5]),self.prevYaw,self.currentPos[2])
      #  self.writeFeet(self.prevFoot,self.footPos,np.array([3]),self.prevYaw,self.currentPos[2])
    else:
      self.prevFoot = copy.deepcopy(self.footPos)
      self.prevYaw = copy.deepcopy(self.currentPos[2])
      self.prevPos = copy.deepcopy(self.currentPos)
      check = self.simulate(self.dd,self.dt,self.dz,tcen)
      # if action == 'b':
      #   self.t = self.t + 10
      global traj_xs
      global traj_ys
      global com_xs
      global com_ys
      global foot_xs
      global foot_ys
      global frame_idxs
      global frame_count
      if self.currentPos[3]==0 and check == 0:
        traj_xs=traj_xs+self.traj_x
        traj_ys=traj_ys+self.traj_y
        com_xs.append(self.prevPos[0])
        com_ys.append(self.prevPos[1])
        foot_xs.append(self.prevFoot[0])
        foot_ys.append(self.prevFoot[1])
        frame_idxs.append(frame_count)
        frame_count=frame_count+len(self.traj_x)
        #self.writeFeet(self.footPos,self.prevFoot,np.array([self.t]),self.currentPos[2],self.prevYaw)
        self.writeFeet(self.footPos,self.currentPos[2]*math.pi/180)
        ang = self.currentPos[2]*math.pi/180+math.pi/2*(1-self.currentPos[3])-math.pi/2*self.currentPos[3]
        self.prevFoot[0:2]=self.currentPos[0:2]-footSpace*np.array([[math.cos(ang)],[math.sin(ang)]])
        self.prevFoot[2]=self.getElevationA(self.prevFoot[0:2]) 
        #self.writeFeet(self.footPos,self.prevFoot,np.array([self.t]),self.currentPos[2],self.currentPos[2])
      elif check == 0:
        traj_xs=traj_xs+self.traj_x
        traj_ys=traj_ys+self.traj_y
        com_xs.append(self.prevPos[0])
        com_ys.append(self.prevPos[1])
        foot_xs.append(self.prevFoot[0])
        foot_ys.append(self.prevFoot[1])
        frame_idxs.append(frame_count)
        frame_count=frame_count+len(self.traj_x)
        #self.writeFeet(self.prevFoot,self.footPos,np.array([self.t]),self.prevYaw,self.currentPos[2])
        self.writeFeet(self.footPos,self.currentPos[2]*math.pi/180)
        ang = self.currentPos[2]*math.pi/180+math.pi/2*(1-self.currentPos[3])-math.pi/2*self.currentPos[3]
        self.prevFoot[0:2]=self.currentPos[0:2]-footSpace*np.array([[math.cos(ang)],[math.sin(ang)]])
        self.prevFoot[2]=self.getElevationA(self.prevFoot[0:2]) 
        #self.writeFeet(self.footPos,self.prevFoot,np.array([self.t]),self.currentPos[2],self.currentPos[2])
      if check == 0:
        self.changeCounter = self.changeCounter + 1
    # Append training samples
    while self.angError > 180:
      self.angError = self.angError - 360
    while self.angError < -180:
      self.angError = self.angError + 360
    Yerror = self.currentPos[2]-targetY
    while Yerror > 180:
      Yerror = Yerror - 360
    while Yerror < -180:
      Yerror = Yerror + 360
    Yerror = np.array([Yerror])
    reg = self.getRegion(self.currentPos) 
    if reg in self.goalids or reg in self.hazardids or reg in self.chargeids:
      self.counter = 0
    elif action == 's' or prevReg == reg:
      pass
    else:
      self.counter = self.counter + 1
    for k in range(1):
      self.Tpos = np.append(self.Tpos,self.footPos[0:2].reshape(2,1),axis=1)
      self.Tval = np.append(self.Tval,self.footPos[2].reshape(-1,1),axis=1)
      self.Mpos = np.append(self.Mpos,np.array([[self.dd],[self.dt.item()],[self.dz.item()],[prevPos[3].item()]]),axis=1)
      #self.Mval = np.append(self.Mval,self.angError.reshape(1,-1)+rng.uniform(-0.5,0.5),axis=1)
      self.Mval = np.append(self.Mval,self.angError.reshape(1,-1),axis=1)
      self.Ypos = np.append(self.Ypos,np.array([[self.zUncertainty.item()],[self.mVar.item()]]),axis=1)
      #self.Yval = np.append(self.Yval,abs(Yerror).reshape(1,-1)+rng.uniform(-0.5,0.5),axis=1)
      self.Yval = np.append(self.Yval,abs(Yerror).reshape(1,-1),axis=1)

  # Method to simulate low-level dynamics
  def simulate(self,dd,dt,dz,tcen):
    #dxdy = self.currentPos[0:2]+dd*np.array([[math.cos(ang)],[math.sin(ang)]])
    tempFootZ = self.footPos[2]
    print("Inputs")
    print(self.vapex)
    print(self.currentPos)
    print(dd)
    print(dt)
    print(dz)
    print(tcen)
    print(self.footPos)
    testFoot = \
      apex_Vel_search(self.currentPos[0].item(), self.currentPos[1].item(), self.zapex, \
      self.waypoint[0].item(), self.waypoint[1].item(), self.vapex, self.footPos[0].item(), self.footPos[1].item(), self.getElevationA(self.footPos[0:2]), \
      self.currentPos[2].item(), dd, dt[0].item(), dz.item(), 1-self.currentPos[3].item())[3]
    if math.isnan(testFoot):
      self.angError=self.yawError(self.dd,self.dt,self.dz,self.footPos[2]-self.footPos[2],tcen[3])
      self.currentPos[2]=self.currentPos[2]+self.dt+self.angError
      self.currentPos[3]=1-self.currentPos[3]
      self.angError = np.array([self.angError])
      self.t = np.array([3])
      return -1
    else: 
      self.currentPos[0],self.currentPos[1],self.zapex, self.footPos[0], \
      self.footPos[1], self.footPos[2], self.vapex, self.t, self.traj_x, self.traj_y = \
      apex_Vel_search(self.currentPos[0].item(), self.currentPos[1].item(), self.zapex, \
      self.waypoint[0].item(), self.waypoint[1].item(), self.vapex, self.footPos[0].item(), self.footPos[1].item(), self.getElevationA(self.footPos[0:2]), \
      self.currentPos[2].item(), dd, dt[0].item(), dz.item(), 1-self.currentPos[3].item())
    print("Outputs")
    self.waypoint=copy.deepcopy(tcen)
    print(self.vapex)
    print(self.footPos[0])
    print(self.footPos[1])
    if self.footPos[0] > gridSize*stateLength:
      self.footPos[0] = gridSize*stateLength
    if self.footPos[1] > gridSize*stateLength:
      self.footPos[1] = gridSize*stateLength
    if self.footPos[0] < 0:
      self.footPos[0] = 0.01
    if self.footPos[1] < 0:
      self.footPos[1] = 0.01
    self.footPos[2]=self.getElevationA(self.footPos[0:2])
    self.currentPos[3]=1-self.currentPos[3]
    # Prevent leaving the boundary of the world
    if self.currentPos[0] > gridSize*stateLength:
      self.currentPos[0] = gridSize*stateLength-.19
    if self.currentPos[1] > gridSize*stateLength:
      self.currentPos[1] = gridSize*stateLength-.19
    if self.currentPos[0] < 0:
      self.currentPos[0] = 0.19
    if self.currentPos[1] < 0:
      self.currentPos[1] = 0.19
    dt=dt+self.yawError(dd,dt,dz,self.footPos[2]-tempFootZ,tcen[3])
    self.currentPos[2] = self.currentPos[2]+dt
    self.currentPos[2]=np.mod(self.currentPos[2],360)
    self.angError = self.currentPos[2]-self.desAng
    return 0

  # Method to simulate yaw error
  def yawError(self,dd,dt,dz,dz_real,fs):
    self.mUnc = self.gpM.calcMean(np.array([[dd],[dt.item()],[dz.item()],[fs.item()]]))
    self.mVar = self.gpM.calcVariance(np.array([[dd],[dt.item()],[dz.item()],[fs.item()]]))
    ye = (dd-stepLength)*10+dt*0.1+(dz-dz_real)*5+self.v.rvs(size=1,random_state=rng.integers(2**32-1))
    ye = ye*meanYawError*.5
    if ye > meanYawError + noiseStd:
      ye = meanYawError + noiseStd
    elif ye < -(meanYawError+noiseStd):
      ye = -(meanYawError+noiseStd)
    return ye-self.mUnc
  
  # Method to train Gaussian processes
  def train(self):
    print("Appending")
    # Add sample to terrain GP
    self.gpT.append(self.Tpos[0:2],self.Tval)
    # Add sample to model GP
    self.gpM.append(self.Mpos,self.Mval)
    # Add sample to yaw GP
    self.gpY.append(self.Ypos,self.Yval)
    # Perform GP Regression
    print("gpT Train")
    self.gpT.train()
    print("gpM Train")
    self.gpM.train()
    print("gpY Train")
    self.gpY.train()
    

  # Method to construct states
  def makeStates(self):
    # Initialize 4800 states in grid world
    states = []
    ids = np.arange(numStates)
    self.goalids = [99]
    self.goals = np.arange(self.goalids[0]*angleLength*2,(self.goalids[0]+1)*angleLength*2).reshape(-1,1)
    #for gid in self.goalids[1:]:
    #  self.goals=np.concatenate((self.goals,np.arange(gid*angleLength*2,(gid+1)*angleLength*2).reshape(-1,1)),axis=1)
    #self.hazardids = np.arange(90,91,2)
    self.hazardids = np.concatenate((np.arange(65,86,20),np.arange(49,51,2)))
    self.hazards = np.arange(self.hazardids[0]*angleLength*2,(self.hazardids[0]+1)*angleLength*2).reshape(-1,1)
    for hid in self.hazardids[1:]:
      self.hazards=np.concatenate((self.hazards,np.arange(hid*angleLength*2,(hid+1)*angleLength*2).reshape(-1,1)),axis=0)
    self.chargeids = np.concatenate((np.arange(0,70,2),np.arange(1,72,3),np.arange(70,99,3),np.array([97,98,87,88,89])))
    self.charges = np.arange(self.chargeids[0]*angleLength*2,(self.chargeids[0]+1)*angleLength*2).reshape(-1,1)
    for cid in self.chargeids[1:]:
      if cid not in self.hazardids:
        self.charges=np.concatenate((self.charges,np.arange(cid*angleLength*2,(cid+1)*angleLength*2).reshape(-1,1)),axis=0)

    for id in ids:
      # Initialize generic states
      tempState = State(id,['l','r','f','b'],[{},{},{},{}],[{},{},{},{}],'u',0)
      tempState.setCenter(self.getCenter(tempState))
      states.append(tempState)
      states[id].setPSat(0)
      # Assign goal and hazard states
      if id in self.goals:
        states[id].setStateType('a')
        states[id].setPSat(1)
        states[id].actions=['s']
      elif id in self.hazards:
        states[id].setStateType('r')
        states[id].actions=['s']
    
    # Duplicate states for multiple steps
    for j in np.arange(1,maxSteps):
      for id in ids:
        tempState = State(id+numStates*j,['l','r','f','b'],[{},{},{},{}],[{},{},{},{}],'u',j)
        tempState.setCenter(self.getCenter(tempState))
        states.append(tempState)
        states[id+numStates*j].setPSat(0)
        if id in self.goals or id in self.charges:
          states[id+numStates*j].setStateType('r')
          states[id+numStates*j].actions=['s']
          self.hazards = np.concatenate((self.hazards,np.array([[id+numStates*j]])),axis=0)
    for id in ids:
      tempState = State(id+numStates*maxSteps,['s'],[{}],[{}],'r',maxSteps)
      tempState.setCenter(self.getCenter(tempState))
      states.append(tempState)
      states[id+numStates*maxSteps].setPSat(0)
      self.hazards = np.concatenate((self.hazards,np.array([[id+numStates*maxSteps]])),axis=0)
      
    # Create State Space
    self.s = StateSpace(states)
    print("Calc Transition Probabilities First")
    self.calcTransitionProbabilitiesFirst()

  # Method to reset states after each iteration
  def resetStates(self):
    ids = np.arange(numStates)
    self.hazards = np.arange(self.hazardids[0]*angleLength*2,(self.hazardids[0]+1)*angleLength*2).reshape(-1,1)
    for hid in self.hazardids[1:]:
      self.hazards=np.concatenate((self.hazards,np.arange(hid*angleLength*2,(hid+1)*angleLength*2).reshape(-1,1)),axis=0)
    for id in ids:
      # Initialize generic states
      self.s.states[id].setPSat(0)
      # Assign goal and hazard states
      if id in self.goals:
        self.s.states[id].setStateType('a')
        self.s.states[id].setPSat(1)
        self.s.states[id].actions=['s']
      elif id in self.hazards:
        self.s.states[id].setStateType('r')
        self.s.states[id].actions=['s']
        self.s.states[id].setPSat(0)
      else:
        self.s.states[id].setPSat(0)
        self.s.states[id].setStateType('u')
    for j in np.arange(1,maxSteps):
      for id in ids:
        self.s.states[id+numStates*j].setPSat(0)
        if id in self.goals or id in self.charges or id in self.hazards:
          self.s.states[id+numStates*j].setStateType('r')
          self.s.states[id+numStates*j].actions=['s']
          self.hazards = np.concatenate((self.hazards,np.array([[id+numStates*j]])),axis=0)
        else:
          self.s.states[id].setPSat(0)
          self.s.states[id].setStateType('u')
    for id in ids:
      self.s.states[id+numStates*maxSteps].setStateType('r')
      self.s.states[id+numStates*maxSteps].actions=['s']
      self.s.states[id+numStates*maxSteps].setPSat(0)
      self.hazards = np.concatenate((self.hazards,np.array([[id+numStates*maxSteps]])),axis=0)
    self.calcTransitionProbabilities()

  # Method to calculate transition probabilities for first iteration
  def calcTransitionProbabilitiesFirst(self):
    # Global variables
    self.dmin = 1000
    self.dmax = 0
    self.zmin = 1000
    self.zmax = 0
    self.thetamin = 1000
    self.thetamax = 0
    # Loop through all states in the state space
    for state in self.s.states:
      id = state.getID()
      if id in self.hazards:
        state.targets = [state.id]
        state.setTarget('s',state.id)
        state.lower=[{}]
        state.upper=[{}]
        state.zminState = 0
        state.zmaxState = 0
        state.zUncertaintyState = 0
        state.dmin = [[0]]
        state.dmax = [[0]]
        state.zmin = [[0]]
        state.zmax = [[0]]
        state.zave = [[0]]
        state.thetamin = [[0]]
        state.thetamax = [[0]]
        state.zUncertainty = [np.array(0).reshape(-1,1)]
      elif id in self.goals:
        state.targets = [state.id]
        state.setTarget('s',state.id)
        state.lower=[{}]
        state.upper=[{}]
        state.zminState = 0
        state.zmaxState = 0
        state.zUncertaintyState = 0
        state.dmin = [[0]]
        state.dmax = [[0]]
        state.zmin = [[0]]
        state.zmax = [[0]]
        state.zave = [[0]]
        state.thetamin = [[0]]
        state.thetamax = [[0]]
        state.zUncertainty = [np.array(0).reshape(-1,1)]
      #print(id)
      else:
        cen = self.getCenter(state)
        # Check if state has invalid foot placement
        ang = cen[2]*math.pi/180+math.pi/2*(1-cen[3])-math.pi/2*cen[3]
        remove = False
        if (not self.checkValidityAlt(cen[0:2]+np.array([[stateLength/2-.05],[stateLength/2-.05]]),cen[0:2]+np.array([[stateLength/2-.05],[stateLength/2-.05]])+footLength*np.array([[math.cos(ang)],[math.sin(ang)]]))) \
          or (not self.checkValidityAlt(cen[0:2]+np.array([[-stateLength/2+.05],[-stateLength/2+.05]]),cen[0:2]+np.array([[-stateLength/2+.05],[-stateLength/2+.05]])+footLength*np.array([[math.cos(ang)],[math.sin(ang)]]))) \
          or (not self.checkValidityAlt(cen[0:2]+np.array([[stateLength/2-.05],[-stateLength/2+.05]]),cen[0:2]+np.array([[stateLength/2-.05],[-stateLength/2+.05]])+footLength*np.array([[math.cos(ang)],[math.sin(ang)]])) )\
          or (not self.checkValidityAlt(cen[0:2]+np.array([[-stateLength/2+.05],[stateLength/2-.05]]),cen[0:2]+np.array([[-stateLength/2+.05],[stateLength/2+.05]])+footLength*np.array([[math.cos(ang)],[math.sin(ang)]]))):
          remove = True
        # Calculate global zmin and zmax for state
        xs = np.linspace(cen[0]-stateLength/2,cen[0]+stateLength/2,num=10)
        ys = np.linspace(cen[1]-stateLength/2,cen[1]+stateLength/2,num=10)
        zmin = 1000
        zmax = 0
        for x in xs:
          for y in ys:
            ztemp = self.gpT.calcMean(np.array([x,y]).reshape(-1,1))
            zUncertaintyTemp = self.gpT.calcVariance(np.array([x,y]).reshape(-1,1))
            if ztemp < zmin:
              zmin = ztemp
            if ztemp > zmax:
              zmax = ztemp
            if zUncertaintyTemp > state.zUncertaintyState:
              state.zUncertaintyState = zUncertaintyTemp
        # Loop through all actions
        toRemove=[]
        removeAct = False
        for action in state.getActions():
          # Calculate desired next state based on action
          target=self.calcTarget(state,action)
          if action =='s':
            continue
          elif target == -1 or remove:
            # Remove invalid actions
            removeAct = True
            toRemove.append(action)
          elif action == 'b' and (not self.checkBackwardsValidity(id)):
              removeAct = True
              toRemove.append(action)
          else:
            state.setTarget(action,target)
            # Calculate GP model parameters
            idx = state.actions.index(action)
            tcen = self.getCenter(self.s.states[target])
            # Calculating range of distances between current state and target
            dx_min = min(abs(tcen[0]-cen[0]-stateLength/2),abs(tcen[0]-cen[0]+stateLength/2))
            dy_min = min(abs(tcen[1]-cen[1]-stateLength/2),abs(tcen[1]-cen[1]+stateLength/2))
            dx_max = max(abs(tcen[0]-cen[0]-stateLength/2),abs(tcen[0]-cen[0]+stateLength/2))
            dy_max = max(abs(tcen[1]-cen[1]-stateLength/2),abs(tcen[1]-cen[1]+stateLength/2))
            state.dmin[idx]=np.sqrt(dx_min ** 2 + dy_min ** 2)
            state.dmax[idx]=np.sqrt(dx_max ** 2 + dy_max ** 2)
            # Calculating range of thetas between current state and target
            state.thetamin[idx]=np.mod(cen[2]-tcen[2]-angleSize,360)
            state.thetamax[idx]=np.mod(cen[2]-tcen[2]+angleSize,360)
            # Calculating range of zs between current state and target
            # Check potential foot placements at target
            tzmin = 1000
            tzmax = 0
            pos2 = tcen[0:2]+footLength*np.array([[math.cos(ang)],[math.sin(ang)]])
            ts = np.linspace(0,1,num=10)
            zUncertaintyAct = state.zUncertaintyState
            for t in ts:
              point = (1-t)*tcen[0:2]+t*pos2
              ztemp = self.gpT.calcMean(point)
              zUncertaintyTemp = self.gpT.calcVariance(point)
              if ztemp < tzmin:
                tzmin = ztemp
              if ztemp > tzmax:
                tzmax = ztemp
              if zUncertaintyTemp > zUncertaintyAct:
                zUncertaintyAct = zUncertaintyTemp
            state.zUncertainty[idx]=zUncertaintyAct
            # Calculate range of dz_foot
            state.zmin[idx] = tzmin-zmax
            state.zmax[idx] = tzmax-zmin
            # Average terrain elevation of next foot placement
            state.zave[idx]=(tzmin+tzmax)/2
            # Calculate global bounds
            if state.dmin[idx] < self.dmin:
              self.dmin = state.dmin[idx]
            if state.dmax[idx] > self.dmax:
              self.dmax = state.dmax[idx]
            if state.zmin[idx] < self.zmin:
              self.zmin = state.zmin[idx]
            if state.zmax[idx] > self.zmax:
              self.zmax = state.zmax[idx]
            if state.thetamin[idx] < self.thetamin:
              self.thetamin = state.thetamin[idx]
            if state.thetamax[idx] > self.thetamax:
              self.thetamax = state.thetamax[idx]
        # Make invalid states hazards
        if removeAct and not remove:
          for act in toRemove:
            state.remove(act)
        if remove or len(state.getActions()) == 0:
          self.hazards = np.concatenate((self.hazards,id.reshape(-1,1)),axis=0)
          state.actions=['s']
          state.lower=[{}]
          state.upper=[{}]
          state.stateType = 'r'
          state.targets = [state.id]
          state.zminState = 0
          state.zmaxState = 0
          state.zUncertaintyState = 0
          state.dmin = [[0]]
          state.dmax = [[0]]
          state.zmin = [[0]]
          state.zmax = [[0]]
          state.zave = [[0]]
          state.thetamin = [[0]]
          state.thetamax = [[0]]
          state.zUncertainty = [np.array(0).reshape(-1,1)]
      # Calculate model error uncertainty globally
    print("Calc Model Uncertainty")
    self.calcModelUncertainty()
    for state in self.s.states:
      for action in state.getActions():
        # Initialize transition probability dictionaries
        lowers = {}
        uppers = {}
        # Calculate reachable states
        yawUncertainty = self.calcYawUncertainty(state,action)
        if state.getID()%1000==24:
          print(yawUncertainty)
        std2y = 2*yawUncertainty
        desired = state.getTarget(action)
        tcen = self.getCenter(self.s.states[desired])
        # Loop through reachable states
        if action=="s":
          lowers={desired:1}
          uppers={desired:1}
        else:
          for y in np.arange(math.floor((tcen[2]-std2y)/angleSize),math.floor((tcen[2]+std2y)/angleSize)+1):
            y = np.mod(y,angleLength)
            # Calculate state id
            counter = self.s.states[desired].getCounter()
            tempID = self.getRegion(tcen[0:2])
            tempID = int(int(tempID*angleLength*2+y*2+tcen[3])+counter*numStates)
            tempCen = self.getCenter(self.s.states[tempID])
            diff = np.mod(tcen[2]-tempCen[2],180)
            # Check for valid state
            if (0<=tempID and tempID<=numStates*(maxSteps+1)):
              # Calculate point in yaw bin to use for lower and upper bounds
              if diff>0:
                lowerY = self.v.cdf(7.5-std2y-diff)-self.v.cdf(-7.5-std2y-diff)
                upperY = self.v.cdf(7.5+std2y-diff)-self.v.cdf(-7.5+std2y-diff)
              elif diff==0:
                lowerY = self.v.cdf(-std2y+7.5)-self.v.cdf(-std2y-7.5)
                upperY = self.v.cdf(7.5)-self.v.cdf(-7.5)
              elif y<0:
                lowerY = self.v.cdf(7.5+std2y-diff)-self.v.cdf(-7.5+std2y-diff)
                upperY = self.v.cdf(7.5-std2y-diff)-self.v.cdf(-7.5-std2y-diff)
              # Add reachable states and transition probabilities
              lowers[tempID] = round(lowerY.item(),2)
              uppers[tempID] = round(upperY.item(),2)
        # Update transition probabilities for each action
        state.update(action,lowers,uppers)
        #if state.getID() % 100 == 24:
        #  state.getInfo()
      #if state.getID()%100==24:
      #  state.getInfo()

  # Method to calculate transition probabilities 
  def calcTransitionProbabilities(self):
    # Loop through all states in the state space
    for state in self.s.states:
      id = state.getID()
      if id in self.hazards or id in self.goals or math.floor(id/numStates) == maxSteps:
        continue
      else:
        #print(id)
        cen = self.getCenter(state)
        ang = cen[2]*math.pi/180+math.pi/2*(1-cen[3])-math.pi/2*cen[3]
        # Calculate global zmin and zmax for state
        xs = np.linspace(cen[0]-stateLength/2,cen[0]+stateLength/2,num=10)
        ys = np.linspace(cen[1]-stateLength/2,cen[1]+stateLength/2,num=10)
        zmin = 1000
        zmax = 0
        for x in xs:
          for y in ys:
            ztemp = self.gpT.calcMean(np.array([x,y]).reshape(-1,1))
            zUncertaintyTemp = self.gpT.calcVariance(np.array([x,y]).reshape(-1,1))
            if ztemp < zmin:
              zmin = ztemp
            if ztemp > zmax:
              zmax = ztemp
            if zUncertaintyTemp > state.zUncertaintyState:
              state.zUncertaintyState = zUncertaintyTemp
        # Loop through all actions
        toRemove=[]
        removeAct = False
        for action in state.getActions():
          # Calculate desired next state based on action
          target=self.calcTarget(state,action)
          if target == -1:
            # Remove invalid actions
            removeAct = True
            toRemove.append(action)
          elif action =='b' and not self.checkBackwardsValidity(id):
              removeAct = True
              toRemove.append(action)
          else:
            state.setTarget(action,target)
            # Calculate GP model parameters
            idx = state.actions.index(action)
            tcen = self.getCenter(self.s.states[target])
            ang = cen[2]*math.pi/180+math.pi/2*(1-cen[3])-math.pi/2*cen[3]
            # Calculating range of distances between current state and target
            # Check potential foot placements at target
            tzmin = 1000
            tzmax = 0
            pos2 = tcen[0:2]+footLength*np.array([[math.cos(ang)],[math.sin(ang)]])
            ts = np.linspace(0,1,num=10)
            zUncertaintyAct = state.zUncertaintyState
            for t in ts:
              point = (1-t)*tcen[0:2]+t*pos2
              ztemp = self.gpT.calcMean(point)
              zUncertaintyTemp = self.gpT.calcVariance(point)
              if ztemp < tzmin:
                tzmin = ztemp
              if ztemp > tzmax:
                tzmax = ztemp
              if zUncertaintyTemp > zUncertaintyAct:
                zUncertaintyAct = zUncertaintyTemp
            state.zUncertainty[idx]=zUncertaintyAct
            # Calculate range of dz_foot
            state.zmin[idx] = tzmin-zmax
            state.zmax[idx] = tzmax-zmin
            # Average terrain elevation of next foot placement
            state.zave[idx]=(tzmin+tzmax)/2
            # Calculate global bounds
            if state.zmin[idx] < self.zmin:
              self.zmin = state.zmin[idx]
            if state.zmax[idx] > self.zmax:
              self.zmax = state.zmax[idx]
        # Make invalid states hazards
        if removeAct:
          for act in toRemove:
            state.remove(act)
        if len(state.getActions()) == 0:
          self.hazards = np.concatenate((self.hazards,id.reshape(-1,1)),axis=0)
          state.actions=['s']
          state.lower=[{}]
          state.upper=[{}]
          state.stateType = 'r'
          state.targets = [state.id]
          state.zminState = 0
          state.zmaxState = 0
          state.zUncertaintyState = 0
          state.dmin = [[0]]
          state.dmax = [[0]]
          state.zmin = [[0]]
          state.zmax = [[0]]
          state.zave = [[0]]
          state.thetamin = [[0]]
          state.thetamax = [[0]]
          state.zUncertainty = [np.array(0).reshape(-1,1)]
    # Calculate model error uncertainty globally
    print("Calc Model Uncertainty")
    self.calcModelUncertainty()
    for state in self.s.states:
      id = state.getID()
      if id in self.hazards or id in self.goals or math.floor(id/numStates) == maxSteps:
        continue
      else:
        for action in state.getActions():
          # Initialize transition probability dictionaries
          lowers = {}
          uppers = {}
          # Calculate reachable states
          yawUncertainty = self.calcYawUncertainty(state,action)
          if state.getID()%1000==24:
            print(yawUncertainty)
          std2y = 2*yawUncertainty
          desired = state.getTarget(action)
          tcen = self.getCenter(self.s.states[desired])
          # Loop through reachable states
          if action=="s":
            lowers={desired:1}
            uppers={desired:1}
          else:
            for y in np.arange(math.floor((tcen[2]-std2y)/angleSize),math.floor((tcen[2]+std2y)/angleSize)+1):
              y = np.mod(y,angleLength)
              # Calculate state id
              counter = self.s.states[desired].getCounter()
              tempID = self.getRegion(tcen[0:2])
              tempID = int(int(tempID*angleLength*2+y*2+tcen[3])+counter*numStates)
              tempCen = self.getCenter(self.s.states[tempID])
              diff = np.mod(tcen[2]-tempCen[2],180)
              # Check for valid state
              if (0<=tempID and tempID<=numStates*(maxSteps+1)):
                # Calculate point in yaw bin to use for lower and upper bounds
                if diff>0:
                  lowerY = self.v.cdf(7.5-std2y-diff)-self.v.cdf(-7.5-std2y-diff)
                  upperY = self.v.cdf(7.5+std2y-diff)-self.v.cdf(-7.5+std2y-diff)
                elif diff==0:
                  lowerY = self.v.cdf(-std2y+7.5)-self.v.cdf(-std2y-7.5)
                  upperY = self.v.cdf(7.5)-self.v.cdf(-7.5)
                elif y<0:
                  lowerY = self.v.cdf(7.5+std2y-diff)-self.v.cdf(-7.5+std2y-diff)
                  upperY = self.v.cdf(7.5-std2y-diff)-self.v.cdf(-7.5-std2y-diff)
                # Add reachable states and transition probabilities
              lowers[tempID] = round(lowerY.item(),2)
              uppers[tempID] = round(upperY.item(),2)
          # Update transition probabilities for each action
          state.update(action,lowers,uppers)
      #state.getInfo()

  # Method to calculate target state for given action and determine validity
  def calcTarget(self,state,action):  
    cen = state.getCenter()
    d = stepLength
    # Calculate desired target destination
    angle = math.pi/180 * cen[2]
    if action == 'l':
      move = np.array([d*math.cos(math.pi/12+angle),d*math.sin(math.pi/12+angle)])
      da=15
    elif action == 'f':
      move = np.array([d*math.cos(angle),d*math.sin(angle)])
      da=0
    elif action == 'r':
      move = np.array([d*math.cos(-math.pi/12+angle),d*math.sin(-math.pi/12+angle)])
      da=-15
    elif action == 'b':
      move = np.array([-d*math.cos(angle),-d*math.sin(angle)])
      da=180
    elif action == 's':
      return state.getID()
    move = move.reshape(-1,1)
    target = np.concatenate((cen[0:2] + move, np.mod(cen[2]+da,360).reshape(-1,1),1-cen[3].reshape(-1,1)),axis=0)
    if not self.checkValidity(cen[0:2],target[0:2]):
      return -1
    counter = state.getCounter()
    id = state.getID()
    nextid = self.getStateID(target)
    if nextid == -1:
      return -1
    if id in self.charges:
      modifier = 1
    else:
      modifier = counter + 1
    if nextid in self.goals or nextid in self.hazards or nextid in self.charges:
      modifier = 0
    if modifier >= maxSteps:
      modifier = maxSteps
    return self.getStateID(target)+numStates*modifier

  # Method to check that robot position/trajectory does not intersect hazard state
  def checkValidity(self,pos1,pos2):
    #if any(pos2<0) or any(pos2>stateLength*gridSize):
    #  return False
    ts = np.linspace(0.0,1.0,num=10)
    for t in ts:
      point = (1-t)*pos1+t*pos2
      reg = self.getRegion(point)
      if reg in self.hazardids:
        return False
    return True

  # Method to check that robot position/trajectory does not intersect hazard state (no out-of-bounds checking)
  def checkValidityAlt(self,pos1,pos2):
    ts = np.linspace(0.0,1.0,num=10)
    for t in ts:
      point = (1-t)*pos1+t*pos2
      reg = self.getRegion(point)
      if reg in self.hazardids:
        return False
    return True

  # Method to check that surrounding states are not hazard regions (for backwards step)
  def checkBackwardsValidity(self,id):
    id = id % numStates
    # Corner cases
    if id==0 and (1 in self.hazardids or 10 in self.hazardids or 11 in self.hazardids):
      #return False
      return True
    elif id==9 and (8 in self.hazardids or 18 in self.hazardids or 19 in self.hazardids):
      #return False
      return True
    elif id==90 and (80 in self.hazardids or 81 in self.hazardids or 91 in self.hazardids):
      #return False
      return True
    # Bottom row
    elif id < 10 and (id-1 in self.hazardids or id+1 in self.hazardids or id+9 in self.hazardids or id+10 in self.hazardids or id+11 in self.hazardids):
      #return False
      return True
    # Top row
    elif id > 89 and (id-1 in self.hazardids or id+1 in self.hazardids or id-9 in self.hazardids or id-10 in self.hazardids or id-11 in self.hazardids):
      #return False
      return True
    # Left column
    if np.mod(id,10)==0 and (id+1 in self.hazardids or id+10 in self.hazardids or id-10 in self.hazardids or id+11 in self.hazardids or id-9 in self.hazardids):
      #return False
      return True
    # Right column
    elif np.mod(id,10)==9 and (id-1 in self.hazardids or id+10 in self.hazardids or id-10 in self.hazardids or id+9 in self.hazardids or id-11 in self.hazardids):
      #return False
      return True
    # Remaining states
    elif id-1 in self.hazardids or id+1 in self.hazardids or id-9 in self.hazardids or id-10 in self.hazardids or id-11 in self.hazardids or id+9 in self.hazardids or id+10 in self.hazardids or id+11 in self.hazardids \
      and id!=0 and id!=9 and id!=90 and id>=10 and id<=89 and np.mod(id,10)!=0 and np.mod(id,10)!=9:
      return False
    # Default
    return True

  # Method to calculate the model uncertainty using global bounds on all of the parameters
  def calcModelUncertainty(self):
    self.ModelUncertainty = np.zeros((10,10,10,2))
    ds = np.linspace(self.dmin,self.dmax,num=10)
    thetas = np.linspace(self.thetamin,self.thetamax,num=10)
    zs = np.linspace(self.zmin,self.zmax,num=10)
    for i in np.arange(0,10):
      for j in np.arange(0,10):
        for k in np.arange(0,10):
          for l in np.arange(0,2):
            self.ModelUncertainty[i,j,k,l] = self.gpM.calcVariance(np.concatenate((ds[i],thetas[j],zs[k,0].reshape(-1),l.reshape(-1))).reshape(4,1))

  # Method to calculate the yaw uncertainty for a given state-action pair
  def calcYawUncertainty(self,state,action):
    idx = state.actions.index(action)
    return self.gpY.calcMean(np.concatenate((state.zUncertainty[idx].reshape(-1,1),\
      np.array([[self.findMaxModelUncertainty(state.dmin[idx],state.dmax[idx],state.thetamin[idx],state.thetamax[idx],state.zmin[idx],state.zmax[idx],state.getCenter()[3])]])),axis=0).reshape(-1,1))

  # Method to calculate maximum model uncertainty given state-action parameters
  def findMaxModelUncertainty(self,dmin,dmax,thetamin,thetamax,zmin,zmax,fs):
    if thetamax < thetamin:
      thetamin = thetamin-360
    didxmin = int(max(np.searchsorted(np.linspace(self.dmin,self.dmax,num=10).reshape(-1),dmin,side='left')-1,0))
    didxmax = int(min(np.searchsorted(np.linspace(self.dmin,self.dmax,num=10).reshape(-1),dmax,side='right')+1,10))
    thetaidxmin = int(max(np.searchsorted(np.linspace(self.thetamin,self.thetamax,num=10).reshape(-1),thetamin,side='left')-1,0))
    thetaidxmax = int(min(np.searchsorted(np.linspace(self.thetamin,self.thetamax,num=10).reshape(-1),thetamax,side='right')+1,10))
    zidxmin = int(max(np.searchsorted(np.linspace(self.zmin,self.zmax,num=10).reshape(-1),zmin,side='left')-1,0))
    zidxmax = int(min(np.searchsorted(np.linspace(self.zmin,self.zmax,num=10).reshape(-1),zmax,side='right')+1,10))
    return np.amax(self.ModelUncertainty[didxmin:didxmax,thetaidxmin:thetaidxmax,zidxmin:zidxmax,int(fs)])

  # Method to calculate safe subgraph
  def pruneGraph(self):
    # Perform max value iteration
    print("Begin Value Iteration Max")
    if self.first == 1:
      self.s.valueIterationMax()
      self.first = 0
    else:
      self.s.valueIterationMaxSimple()
    print("End Value Iteration Max")
    if hasattr(self, 'safeSG'):
      del self.safeSG
    # Create copy of graph to prune
    self.safeSG = copy.deepcopy(self.s)
    # Identify states to prune
    self.hazardstemp = []
    for state in self.safeSG.states:
      if state.getPSat() == 0:
        self.hazardstemp.append(state.getID())
        self.hazards = np.concatenate((self.hazards,state.getID().reshape(-1,1)),axis=0)
    for id in self.hazardstemp:
      idx = self.safeSG.ids.index(id)
      self.safeSG.remove(self.safeSG.states[idx])
      #print(id)
    #Loop while more states have been added as hazards
    newStates = 1
    while newStates != 0:
      newStates = 0
      # Loop through states
      toRemove = []
      for state in self.safeSG.states:
        # Loop through actions
        toRemoveAct = []
        for action in state.getActions():
          # Get reachable states
          next = state.getNext(action)
          # Check if action transitions to hazard state
          if not (set(next).isdisjoint(self.hazardstemp)):
            #print(str(state.getID())+" "+str(action)+" ")
            #print(next)
            # Remove action
            toRemoveAct.append(action)
        # Remove actions
        for act in toRemoveAct:
          state.remove(act)
        # Check if action set is empty
        if len(state.getActions()) == 0:
          # Prune state and treat it as a hazard
          toRemove.append(state.getID())
          self.hazardstemp.append(state.getID())
          self.hazards = np.concatenate((self.hazards,state.getID().reshape(-1,1)),axis=0)
          # Remove redundant hazards
          self.hazardstemp = list(set(self.hazardstemp))
          newStates = 1
      for id in toRemove:
        idx = self.safeSG.ids.index(id)
        self.safeSG.remove(self.safeSG.states[idx])
    print("Done Prune Graph")
    #print(self.hazardstemp)
    #print(self.safeSG.ids)
    #self.safeSG.getInfo()

  # Method to synthesize control policy given calculated optimal MEC
  def synthesize(self):
    self.policy = {}
    # Perform reachability calculations to optimal MEC
    # Create a copy of the state space to manipulate
    print("Pre Deep Copy")
    if hasattr(self, 'tempG'):
      del self.tempG
    self.tempG = copy.deepcopy(self.s)
    # Loop through states and reassign probability of satisfaction (1 and absorbing if in MEC, 0 otherwise)
    print("Deep Copy")
    for state in self.tempG.states:
      if state.getID() in self.safeSG.ids:
        state.actions = ['s']
        state.lower = [{state.getID():1}]
        state.upper = [{state.getID():1}]
        state.setPSat(1)
        state.setStateType('a')
      else:
        state.setPSat(0)
        state.setStateType('u')
    # Calculate reachability probabilities from initial state
    self.tempG.valueIterationMin()
    print("Value Iteration Synthesize")
    # Loop through states
    for state in self.s.states:
      # Check if state is in MEC
      if state.getID() in self.safeSG.ids:
        # Append all valid actions in MEC
        idx = self.safeSG.ids.index(state.getID())
        self.policy[state.getID()]=self.safeSG.states[idx].getActions()
      else:
        # Append optimal action
        idx = self.tempG.ids.index(state.getID())
        self.policy[state.getID()]=self.tempG.states[idx].getOptAct()
    #print(self.policy)
        
  # Method to reach goal on final iteration      
  def reachGoal(self):
    self.policy = {}
    self.s.valueIterationMin()
    print("Value Iteration Synthesize")
    # Loop through states
    for state in self.s.states:
      self.policy[state.getID()]=state.getOptAct()

  # Method to calculate total transition uncertainty
  def calcUncertainty(self):
    # Loop through states
    tempUncertainty = 0
    for state in self.s.states:
      # Loop through actions
      for idx in range(len(state.actions)):
        # Extract lower and upper transition probabilities
        tempLower = state.lower[idx]
        tempUpper = state.upper[idx]
        tempLower = np.asarray(list(tempLower.values()))
        tempUpper = np.asarray(list(tempUpper.values()))
        # Sum and append to array
        tempUncertainty = tempUncertainty + np.sum(tempUpper-tempLower)
    tempUncertainty = tempUncertainty.reshape(-1,1)
    self.uncertainty=np.append(self.uncertainty,tempUncertainty,axis=1)

  # Method to calculate terrain error between unknown dynamics and GP
  def calcError(self):
    # Loop through states
    tempError = 0
    for state in self.s.states:
      # Extract estimated and true dynamics and calculate error
      errorZ = np.abs(self.gpT.calcMean(self.getCenter(state)[0:2])-self.getElevationA(self.getCenter(state)[0:2]))
      tempError = tempError + errorZ
    # Append error to history vector
    self.errors = np.append(self.errors,np.asarray(tempError).reshape(-1,1))
  
  # Method to write foot placements for trajectory visualization
  def writeFeet(self,foot,yaw):
    yaw = np.mod(yaw,2*math.pi)
    self.writer.writerow([foot[0].item(),foot[1].item(),yaw.item()])

   # Method to write foot placements for trajectory visualization
  # def writeFeet(self,left,right,time,leftYaw,rightYaw):
  #   self.time = self.time + time
  #   self.file1.write('{"foot_constraints": {"LF": {"name": "LF", "xyz": ['+str(left[0,0])+','+str(left[1,0])+','+str(left[2,0])+'],'+ \
  #     '"rpy": [0, 0, '+str(leftYaw.item()*math.pi/180)+']}, "RF": {"name": "RF", "xyz": ['+str(right[0,0])+','+str(right[1,0])+','+str(right[2,0])+'],'+ \
  #     '"rpy": [0, 0, '+str(rightYaw.item()*math.pi/180)+ ']}}, "link_constraints": {}, "enable_com_constraint": [true, true], "time": ' + str(self.time.item())+', "mode": 0},')
  
  # Final method to run entire algorithm
  def run(self):
    self.first = 1
    print("Starting")
    # Construct states
    self.makeStates()
    # Calculate uncertainty
    self.calcUncertainty()
    #self.s.getInfo()
    # Calculate error
    #self.calcError()
    #print("CalcError")
    # Perform min reachability probability value iteration
    self.s.valueIterationMin()
    print("Value Iteration Min")
    psat = self.s.states[self.initState].getPSat()
    self.counter = self.s.states[self.initState].getCounter()
    self.psats = np.append(self.psats,psat.reshape(-1,1),axis=1)
    print("P_sat: " + str(psat))
    numIterOut = 20
    numIterIn = 18
    counterOut = 0
    self.pos = np.empty((2,0))
    self.fpos = np.empty((2,0))
     # Open file to write
    self.file1 = open('jesse_ms2.csv', 'a')
    self.writer = csv.writer(self.file1)
    #self.file1.write("[")
    f_d = math.sqrt((self.currentPos[0]-self.footPos[0])**2 + (self.currentPos[1]-self.footPos[1])**2)
    ang = self.currentPos[2]*math.pi/180+math.pi/2*(1-self.currentPos[3])-math.pi/2*self.currentPos[3]
    dxdy = self.currentPos[0:2]+f_d*np.array([[math.cos(ang)],[math.sin(ang)]])
    self.footPos[0]=dxdy[0]
    self.footPos[1]=dxdy[1]
    self.footPos[2]=self.getElevationA(self.footPos[0:2])
    # self.writeFeet(self.footPos,np.array([[(self.footPos[0]+2*(self.currentPos[0]-self.footPos[0])).item()], \
    #   [(self.footPos[1]+2*(self.currentPos[1]-self.footPos[1])).item()],[self.footPos[2].item()]]),np.array([0]),self.currentPos[2],self.currentPos[2])
    # Iterate while successful control policy has not been calculated and max number of iterations has not been reached
    self.writeFeet(np.array([[(self.footPos[0]+2*(self.currentPos[0]-self.footPos[0])).item()],[(self.footPos[1]+2*(self.currentPos[1]-self.footPos[1])).item()],[self.footPos[2].item()]]),self.currentPos[2]*math.pi/180)
    self.writeFeet(self.footPos,self.currentPos[2]*math.pi/180)
    while (counterOut < numIterOut):# and (psat<self.Psat):
      print(["Iteration: " + str(counterOut)])
      # Calculate safe subgraph
      self.pruneGraph()
      # Calculate optimal MEC
      print("Prune Graph")
      #self.calcMECs()
      #print(self.mecs)
      #print("Calc MECs")
      #self.calcOptimalMEC()
      #print("Calc Optimal MEC")
      #print(self.optimalMEC)
      # Synthesize control policy
      # Synthesize control policy
      if psat > .99999:
        self.reachGoal()
      else:
        self.synthesize()
      #self.s.getInfo()
      # print("Policy:")
      # print(self.policy)
      # Initialize finite memory controller
      print("Done Synthesizing")
      tracker = np.zeros((numStates*maxSteps))
      # Follow control policy for max number of iterations
      counterIn = 0
      # Reset apex velocity
      self.vapex = 0.35
      done = False
      self.changeCounter = 0
      while self.changeCounter < numIterIn:
        if (self.getRegion(self.currentPos[0:2]) in self.goalids) or (self.getRegion(self.currentPos[0:2]) in self.hazardids):
          print("Done/Hazard")
          done = True
          self.pos = np.concatenate((self.pos,self.currentPos[0:2]),axis=1)
          self.fpos = np.concatenate((self.fpos,self.footPos[0:2]),axis=1)
          break
        if self.counter == 3:
          done = True
          print("Failed")
          break
        self.pos = np.concatenate((self.pos,self.currentPos[0:2]),axis=1)
        self.fpos = np.concatenate((self.fpos,self.footPos[0:2]),axis=1)
        #print(self.currentPos)
        # Step in the state space
        self.currState = self.getStateID(self.currentPos) + numStates * self.counter
        #print(self.currState)
        # Calculate action using finite memory controller
        action = self.policy[self.currState][rng.integers(0,len(self.policy[self.currState]))]
        #tracker[self.currState]=tracker[self.currState]+1
        # Calculate desired next state based on action
        self.step(self.s.states[self.currState],action)
        print(self.currentPos)
        print(action)
        print(self.getRegion(self.currentPos[0:2]))
        print(self.counter)
        print()
        #nextid = self.s.states[self.getStateID(self.currentPos)]
        counterIn = counterIn + 1
        self.stepCounter = self.stepCounter + 1
      if (self.getRegion(self.currentPos[0:2]) in self.goalids):
        done = True
        self.pos = np.concatenate((self.pos,self.currentPos[0:2]),axis=1)
        self.fpos = np.concatenate((self.fpos,self.footPos[0:2]),axis=1)
        self.stepCounter = self.stepCounter + 1
      if done:
        self.pos = np.concatenate((self.pos,self.currentPos[0:2]),axis=1)
        self.fpos = np.concatenate((self.fpos,self.footPos[0:2]),axis=1)
        break
      print("Training")
      # Train Gaussian Process
      self.train()
      print(self.gpY.val)
      print("Reset States")
      # Reconstruct transition probabilities
      self.resetStates()
      #print("Calculating Uncertainty")
      # Calculate uncertainty
      self.calcUncertainty()
      # Calculate error
      #print("Calculating Error")
      #self.calcError()
      print("Value Iteration Min")
      # first = 1
      # if first == 1: 
      #   self.s.valueIterationMin()
      #   first = 0
      # else:
      if psat < 0.001:
        self.s.valueIterationMin()
        numIterIn = 18
      else:
        self.s.valueIterationMinSimple()
        numIterIn = 18
      # Calculate new probability of satisfying the specification
      psat = self.s.states[self.currState].getPSat()
      self.psats = np.append(self.psats,psat.reshape(-1,1),axis=1)
      print("P_sat: " + str(psat))
      self.s.getOptActs()
      print("Optimal Actions:")
      #print(self.s.optActs)
      counterOut = counterOut + 1
      self.stepIter.append(self.stepCounter)
    #self.file1.write("]")
    self.file1.close()
    self.stepCounter = self.stepCounter + 1
    self.stepIter.append(self.stepCounter)
    print(["Iteration: " + str(counterOut)])
    # Calculate safe subgraph
    #self.pruneGraph()
    # Calculate optimal MEC
    #self.calcMECs()
    #self.calcOptimalMEC()
    #print(self.optimalMEC)
    # for state in self.s.states:
    #   print("Estimated terrain: " + str(self.gpT.calcMean(self.getCenter(state)[0:2])))
    #   print("Estimated terrain uncertainty: " + str(self.gpT.calcVariance(self.getCenter(state)[0:2])))
    #   print("Actual terrain: " + str(self.gT.calcMean(self.getCenter(state)[0:2])))
    #   state.getInfo()
    # Animation
    # print("Frame Count")
    # print(frame_count)
    # ani = animation.FuncAnimation(
    #   f3, animate, math.ceil(frame_count/compression), interval=1, blit=True)
    # writer = animation.writers['ffmpeg'](fps=30)
    # ani.save('multistep_casestudy517.mp4',writer=writer,dpi=500)
    return counterOut


# %% [markdown]
#  Run Case Study
# 

# %%

# %%
import jax.numpy as jnp
import os
os.environ['TF_FORCE_UNIFIED_MEMORY'] = '1'
try:
    jnp.linalg.qr(jnp.array([[0, 1], [1, 1]]))
except RuntimeError:
    pass
# Testing GridWorld state construction
print("Prestarting")
import time
msg = MultiStepGoal()
N = 1
iterations = np.empty((N))
times = np.empty((N))
uncertainties = []
psats = []
tempPos = np.empty((2,0))
tempVal = np.empty((2,0))
for i in np.arange(N):
    start = time.time()
    iterations[i]=msg.run()
    end = time.time()
    print("Time Elapsed: " + str(end-start))
    times[i]=end-start
    uncertainties.append(msg.uncertainty[0])
    psats.append(msg.psats[0])
    tempPos = msg.pos
    msg.uncertainty = np.empty((1,0))
    msg.psats = np.empty((1,0))
    msg.gpT.pos=np.empty((2,0))
    msg.gpT.val=np.empty((1,0))
    msg.gpM.pos=np.empty((4,0))
    msg.gpM.val=np.empty((1,0))
    msg.gpY.pos=np.empty((2,0))
    msg.gpY.val=np.empty((1,0))
    msg.Tpos = np.empty((2,0))
    msg.Tval = np.empty((1,0))
    msg.Mpos = np.empty((4,0))
    msg.Mval = np.empty((1,0))
    msg.Ypos = np.empty((2,0))
    msg.Yval = np.empty((1,0))
print("Average Iterations: " + str(np.average(iterations)))
print("Average Time: " + str(np.average(times)))


# %% [markdown]
# 
# 
# Plots
# 

# %%

# %%
# Plot results
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
f1 = plt.figure()
fig, ax = plt.subplots(nrows=1, ncols=1)
fig.set_size_inches(16, 7)
c=0
    #if c==0:
for i in np.arange(len(uncertainties)):
            ax.plot(np.arange(np.size(uncertainties[i],0)),uncertainties[i])
            ax.set_title("Transition Uncertainty (Case Study 2)")
            ax.set_xlabel("Iteration #")
            ax.set_ylabel("Total Uncertainty")
            ax.set_xticks(np.arange(0,iterations[0]+1,step=1))
    # if c==1:
    #     for i in np.arange(len(psats)):
    #         col.plot(np.arange(np.size(psats[i],0)),psats[i])
    #         col.set_title("Prob. Satisfaction")
    #         col.set_xlabel("Iteration #")
    #         col.set_ylabel("Probability")
    #c=1
fig.tight_layout()
for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
              ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(26)
fig.savefig("yaw_uncertainties_multistep.pdf", bbox_inches='tight')
plt.show()
#Plot error of unknown dynamics estimation
# plt.plot(gw.errors)
# plt.title("Error of Gaussian Process Estimation of Unknown Dynamics")
# plt.xlabel("Iteration #")
# plt.ylabel("Total Error")
# plt.figure()

# %%

# Plot trajectory
img2 = plt.imread("legged_robot_multistep_casestudy.png")
f2 = plt.figure()
plt.imshow(img2,extent=[0,1.5,0,1.5])
dim = np.size(tempPos,1)
plt.xticks(np.arange(0, gridSize*stepLength+stepLength, step=stepLength))
plt.yticks(np.arange(0, gridSize*stepLength+stepLength, step=stepLength))
for i in np.arange(len(msg.stepIter)-1):
  plt.plot(tempPos[0,msg.stepIter[i]:msg.stepIter[i+1]+1],tempPos[1,msg.stepIter[i]:msg.stepIter[i+1]+1],alpha=0.6, marker='o', label=str(i+1))
  plt.scatter(msg.fpos[0,msg.stepIter[i]:msg.stepIter[i+1]+1],msg.fpos[1,msg.stepIter[i]:msg.stepIter[i+1]+1],c="cyan",alpha=0.4,marker='*')
plt.legend(loc='upper left', fancybox=True, title='Batch #')
plt.show()
f2.tight_layout()
f2.savefig("legged_robot_multistep_casestudy_new.png", bbox_inches='tight', dpi=1000)


# %%

# %%
import jax
print(jax.devices())



