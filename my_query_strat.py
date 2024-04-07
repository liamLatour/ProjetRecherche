import numpy as np
from sklearn.base import clone

from modAL.models import ActiveLearner
from modAL.uncertainty import _proba_entropy, _proba_uncertainty
from modAL.utils.data import (add_row, data_shape, data_vstack, drop_rows,
                              enumerate_data, modALinput)
from modAL.utils.selection import multi_argmin, multi_argmax

import itertools
from tqdm import tqdm

from scipy.optimize import minimize, shgo, dual_annealing
from scipy.special import kl_div

import warnings
warnings.filterwarnings("error")

def eer_single(learner: ActiveLearner, x_samples: modALinput, n_instances: int = 1):
  expected_error = np.zeros(shape=(data_shape(x_samples)[0],))
  possible_labels = np.unique(learner.y_training)

  x_proba = learner.predict_proba(x_samples)
  cloned_estimator = clone(learner.estimator)
  
  warns = []
  
  for x_idx, x in tqdm(enumerate_data(x_samples), total=len(x_samples)):
    x_reduced = drop_rows(x_samples, x_idx)
    # estimate the expected error
    for y_idx, y in enumerate(possible_labels):
      x_new = add_row(learner.X_training, x)
      y_new = data_vstack((learner.y_training, np.array(y).reshape(1,)))

      with warnings.catch_warnings():
        warnings.filterwarnings('error')
        try:
          cloned_estimator.fit(x_new, y_new)
          refitted_proba = cloned_estimator.predict_proba(x_reduced)
          #nloss = _proba_uncertainty(refitted_proba)
          nloss = _proba_entropy(refitted_proba)
        except Warning:
          warns.append(x)
          nloss = 100000000
      
      expected_error[x_idx] += np.sum(nloss)*x_proba[x_idx, y_idx]

  best = multi_argmin(expected_error, n_instances)

  return (best[1], x_samples[best[0]], warns)

def emc_single(learner: ActiveLearner, x_samples: modALinput, n_instances: int = 1):
  expected_error = np.zeros(shape=(data_shape(x_samples)[0],))
  possible_labels = np.unique(learner.y_training)

  x_proba = learner.predict_proba(x_samples)
  cloned_estimator = clone(learner.estimator)
  
  params = learner.estimator.kernel_.get_params(deep=True)
  initial_k1 = params['k1__constant_value']
  initial_k2 = params['k2__length_scale']
  
  warns = []
  
  for x_idx, x in tqdm(enumerate_data(x_samples), total=len(x_samples)):
    # estimate the expected error
    for y_idx, y in enumerate(possible_labels):
      x_new = add_row(learner.X_training, x)
      y_new = data_vstack((learner.y_training, np.array(y).reshape(1,)))

      with warnings.catch_warnings():
        warnings.filterwarnings('error')
        try:
          cloned_estimator.fit(x_new, y_new)
          params = cloned_estimator.kernel_.get_params(deep=True)
          nchange = (initial_k1 - params['k1__constant_value'])**2 + (initial_k2 - params['k2__length_scale'])**2
        except Warning:
          warns.append(x)
          nchange = 0

      expected_error[x_idx] += nchange*x_proba[x_idx, y_idx]

  best = multi_argmax(expected_error, n_instances)

  return (best[1], x_samples[best[0]], warns)

def eer_multi(learner: ActiveLearner, x_samples: modALinput, n_instances: int = 1) -> np.ndarray:
    possible_labels = np.unique(learner.y_training)

    x_proba = learner.predict_proba(x_samples)
    cloned_estimator = clone(learner.estimator)
    
    # all combinations of n_instances samples
    combinations_x = list(itertools.combinations(x_samples, n_instances))
    combinations_x_idx = list(itertools.combinations(range(len(x_samples)), n_instances))
    combinations_x = list(zip(combinations_x, combinations_x_idx))
    
    # all combinations of possible_labels
    combinations_y = list(itertools.product(possible_labels, repeat=n_instances))
    
    expected_error = np.zeros(shape=(len(combinations_x),))
    
    for x_idx, x in tqdm(enumerate_data(combinations_x), total=len(combinations_x)):
      x_reduced = drop_rows(x_samples, x[1])
      # estimate the expected error
      for y in combinations_y:
        proba = 1
        
        for i in range(n_instances):
          proba *= x_proba[x[1][i], y[i]]
          
        #if proba > 0.1:
        x_new = add_row(learner.X_training, x[0])
        y_new = data_vstack((learner.y_training, y))
        
        cloned_estimator.fit(x_new, y_new)
        refitted_proba = cloned_estimator.predict_proba(x_reduced)
        #nloss = _proba_uncertainty(refitted_proba) # binary
        nloss = _proba_entropy(refitted_proba) # log

        expected_error[x_idx] += np.sum(nloss)*proba

    best = multi_argmin(expected_error, 1)
    
    return (np.asarray(combinations_x_idx[best[0][0]]), np.ones((n_instances))*best[1][0])

def emc_multi(learner: ActiveLearner, x_samples: modALinput, n_instances: int = 1) -> np.ndarray:
    possible_labels = np.unique(learner.y_training)

    x_proba = learner.predict_proba(x_samples)
    params = learner.estimator.kernel_.get_params(deep=True)
    initial_k1 = params['k1__constant_value']
    initial_k2 = params['k2__length_scale']

    cloned_estimator = clone(learner.estimator)
    
    # all combinations of n_instances samples
    combinations_x = list(itertools.combinations(x_samples, n_instances))
    combinations_x_idx = list(itertools.combinations(range(len(x_samples)), n_instances))
    combinations_x = list(zip(combinations_x, combinations_x_idx))
    
    # all combinations of possible_labels
    combinations_y = list(itertools.product(possible_labels, repeat=n_instances))
    
    expected_change = np.zeros(shape=(len(combinations_x),))
    
    for x_idx, x in tqdm(enumerate_data(combinations_x), total=len(combinations_x)):
      for y in combinations_y:
        proba = 1
        
        for i in range(n_instances):
          proba *= x_proba[x[1][i], y[i]]
          
        #if proba > 0.1:
        x_new = add_row(learner.X_training, x[0])
        y_new = data_vstack((learner.y_training, y))
        
        cloned_estimator.fit(x_new, y_new)

        params = cloned_estimator.kernel_.get_params(deep=True)
        nchange = (initial_k1 - params['k1__constant_value'])**2 + (initial_k2 - params['k2__length_scale'])**2

        expected_change[x_idx] += nchange*proba

    best = multi_argmax(expected_change, 1)
    
    return (np.asarray(combinations_x_idx[best[0][0]]), np.ones((n_instances))*best[1][0])

def emc_continuous(model, n_instances: int = 1) -> np.ndarray:
    learner = model.learner
    possible_labels = np.unique(learner.y_training)

    #x_proba = learner.predict_proba(x_samples)
    params = learner.estimator.kernel_.get_params(deep=True)
    initial_k1 = params['k1__constant_value']
    initial_k2 = params['k2__length_scale']

    cloned_estimator = clone(learner.estimator)
    
    # all combinations of n_instances samples
    #combinations_x = list(itertools.combinations(x_samples, n_instances))
    #combinations_x_idx = list(itertools.combinations(range(len(x_samples)), n_instances))
    #combinations_x = list(zip(combinations_x, combinations_x_idx))
    
    # all combinations of possible_labels
    combinations_y = list(itertools.product(possible_labels, repeat=n_instances))
    
    def valuation(data):
      #print(data.reshape((-1,2)))
      points = model.scale(data.reshape((-1,2)))
      x_proba = learner.predict_proba(points)
      expected_change = 0
      
      for y in combinations_y:
        proba = 1
        
        for i in range(n_instances):
          proba *= x_proba[i, y[i]]
      
        x_new = add_row(learner.X_training, points)
        y_new = data_vstack((learner.y_training, y))
        
        cloned_estimator.fit(x_new, y_new)
        
        params = cloned_estimator.kernel_.get_params(deep=True)
        nchange = (initial_k1 - params['k1__constant_value'])**2 + (initial_k2 - params['k2__length_scale'])**2
    
        expected_change += nchange*proba
        
      #print(expected_change)
      return -expected_change
    
    bounds = tuple(((0,100) for _ in range(n_instances*2)))
    
    # best = minimize(valuation,
    #                 np.array([30 for _ in range(n_instances*2)]),
    #                 method='BFGS',
    #                 options={'eps': 5})
                    #bounds=bounds)
    
    best = dual_annealing(valuation, bounds)
    
    print(best)
    print(best.x)
    
    return best.x.reshape((-1,2)) #(np.asarray(combinations_x_idx[best[0][0]]), np.ones((n_instances))*best[1][0])
