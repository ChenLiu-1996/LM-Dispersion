from typing import List
import torch
import numpy as np


def apply_diffusion_operator(X: torch.Tensor,
                             t: int = 20) -> List[torch.Tensor]:
    '''
    `X` : [N, D] feature matrix,
        where N := number of feature vectors
              D := number of features
    '''
    P = _compute_operator(X)

    output_list = []
    for power in range(1, t + 1):
        curr_P = torch.linalg.matrix_power(P, power)
        output_list.append(curr_P @ X)

    return output_list


def _compute_operator(X: torch.Tensor,
                      sigma: float = 1.0) -> torch.Tensor:
    '''
    Compute the diffusion operator.
    '''
    # Compute squared distance matrix.
    dist2 = torch.cdist(X, X, p=2) ** 2
    G = torch.exp(-dist2 / (2 * sigma ** 2))

    # Normalize rows to form a Markov matrix
    row_sums = G.sum(dim=1, keepdim=True)
    P = G / (row_sums + 1e-9)

    stay_prob = 0.9
    P = torch.eye(P.shape[0], device=P.device) * stay_prob + P * (1 - stay_prob)
    return P