import torch
import numpy as np
from typing import List

from .utils import fix_random_seed
from sklearn.cluster import KMeans

def random_sampling(pred_probs: torch.Tensor, budget : int = 1000) -> List[int]:
  '''Randomly pick examples.
  :param pred_probs: list of predicted probabilities for the production set in order.
  :param budget: the number of examples you are allowed to pick for labeling.
  :return indices: A list of indices (into the `pred_probs`) for examples to label.
  '''
  fix_random_seed(42)
  
  indices = []
  # ================================
  # FILL ME OUT
  # Randomly pick a 1000 examples to label. This serves as a baseline.
  # Note that we fixed the random seed above. Please do not edit.
  # HINT: when you randomly sample, do not choose duplicates.
  # HINT: please ensure indices is a list of integers
  # ================================
  indices = np.random.choice(
    np.arange(len(pred_probs)), 
    budget, 
    replace=False
  )
  indices.sort()
  indices = indices.tolist()
  return indices

def uncertainty_sampling(pred_probs: torch.Tensor, budget : int = 1000) -> List[int]:
  '''Pick examples where the model is the least confident in its predictions.
  :param pred_probs: list of predicted probabilities for the production set in order.
  :param budget: the number of examples you are allowed to pick for labeling.
  :return indices: A list of indices (into the `pred_probs`) for examples to label.
  '''
  indices = []
  chance_prob = 1 / 10.  # may be useful
  # ================================
  # FILL ME OUT
  # Sort indices by the predicted probabilities and choose the 1000 examples with 
  # the least confident predictions. Think carefully about what "least confident" means 
  # for a N-way classification problem.
  # Take the first 1000.
  # HINT: please ensure indices is a list of integers
  # ================================
  
  # The least confident examples are those where the predicted probability is closest to the chance probability
  # Each prediction is a vector of probabilities for each class. We can calculate the distance between the predicted
  # probability and the chance probability for each class. The sum of these distances will be the distance between the
  # predicted probability and the chance probability for the entire vector. We can sort the examples by this distance
  # and take the 1000 examples with the smallest distances.
  distances = np.sum(np.abs(pred_probs.numpy() - chance_prob), axis=1)
  indices = np.argsort(distances)[:budget]
  indices.sort()
  indices = indices.tolist()

  return indices

def margin_sampling(pred_probs: torch.Tensor, budget : int = 1000) -> List[int]:
  '''Pick examples where the difference between the top two predicted probabilities is the smallest.
  :param pred_probs: list of predicted probabilities for the production set in order.
  :param budget: the number of examples you are allowed to pick for labeling.
  :return indices: A list of indices (into the `pred_probs`) for examples to label.
  '''
  indices = []
  # ================================
  # FILL ME OUT
  # Sort indices by the different in predicted probabilities in the top two classes per example.
  # Take the first 1000.
  # ================================
  probs_sorted = np.sort(pred_probs.numpy(), axis=1)
  top_two_abs_diff = np.abs(probs_sorted[:, -1] - probs_sorted[:, -2])
  indices = np.argsort(top_two_abs_diff)[:budget].tolist()
  
  return indices

def entropy_sampling(pred_probs: torch.Tensor, budget : int = 1000) -> List[int]:
  '''Pick examples with the highest entropy in the predicted probabilities.
  :param pred_probs: list of predicted probabilities for the production set in order.
  :param budget: the number of examples you are allowed to pick for labeling.
  :return indices: A list of indices (into the `pred_probs`) for examples to label.
  '''
  indices = []
  epsilon = 1e-6
  # ================================
  # FILL ME OUT
  # Entropy is defined as -E_classes[log p(class | input)] aja the expected log probability
  # over all K classes. See https://en.wikipedia.org/wiki/Entropy_(information_theory).
  # Sort the indices by the entropy of the predicted probabilities from high to low.
  # Take the first 1000.
  # HINT: Add epsilon when taking a log for entropy computation
  # ================================
  # Math definition: -sum_i (p_i * log(p_i))
  # Add epsilon to prevent log(0) errors
  probs = pred_probs.numpy()
  entropy = -np.sum(probs * np.log(probs + epsilon))
  indices = np.argsort(-entropy)[:budget].tolist()

  return indices
