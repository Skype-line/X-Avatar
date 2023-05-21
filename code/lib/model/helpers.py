import torch
import cv2
import numpy as np


def masked_softmax(vec, mask, dim=-1, mode='softmax', soft_blend=1):
    if mode == 'softmax':
        vec = torch.distributions.Bernoulli(logits=vec).probs

        masked_exps = torch.exp(soft_blend*vec) * mask.float()
        masked_exps_sum = masked_exps.sum(dim)

        output = torch.zeros_like(vec)
        output[masked_exps_sum>0,:] = masked_exps[masked_exps_sum>0,:]/ masked_exps_sum[masked_exps_sum>0].unsqueeze(-1)

        output = (output * vec).sum(dim, keepdim=True)

        value = torch.distributions.Bernoulli(probs=output).logits
        index = None
    elif mode == 'max':
        vec[~mask.bool()] = -100
        value, index = torch.max(vec, dim, keepdim=True)
    elif mode == 'min':
        vec[~mask.bool()] = 1
        value, index = torch.min(vec, dim, keepdim=True)
    else:
        raise NotImplementedError
    return value, index


''' Hierarchical softmax following the kinematic tree of the human body. Imporves convergence speed'''
def hierarchical_softmax(x, model_type):
    def softmax(x):
        return torch.nn.functional.softmax(x, dim=-1)

    def sigmoid(x):
        return torch.sigmoid(x)

    n_batch, n_point, n_dim = x.shape
    x = x.flatten(0,1)

    if model_type == 'smpl':
        prob_all = torch.ones(n_batch * n_point, n_dim-1, device=x.device)
        
        prob_all[:, [1, 2, 3]] = prob_all[:, [0]] * sigmoid(x[:, [0]]) * softmax(x[:, [1, 2, 3]])
        prob_all[:, [0]] = prob_all[:, [0]] * (1 - sigmoid(x[:, [0]]))

        prob_all[:, [4, 5, 6]] = prob_all[:, [1, 2, 3]] * (sigmoid(x[:, [4, 5, 6]]))
        prob_all[:, [1, 2, 3]] = prob_all[:, [1, 2, 3]] * (1 - sigmoid(x[:, [4, 5, 6]]))

        prob_all[:, [7, 8, 9]] = prob_all[:, [4, 5, 6]] * (sigmoid(x[:, [7, 8, 9]]))
        prob_all[:, [4, 5, 6]] = prob_all[:, [4, 5, 6]] * (1 - sigmoid(x[:, [7, 8, 9]]))

        prob_all[:, [10, 11]] = prob_all[:, [7, 8]] * (sigmoid(x[:, [10, 11]]))
        prob_all[:, [7, 8]] = prob_all[:, [7, 8]] * (1 - sigmoid(x[:, [10, 11]]))
        
        prob_all[:, [12, 13, 14]] = prob_all[:, [9]] * sigmoid(x[:, [n_dim-1]]) * softmax(x[:, [12, 13, 14]])
        prob_all[:, [9]] = prob_all[:, [9]] * (1 - sigmoid(x[:, [n_dim-1]]))
        
        prob_all[:, [15]] = prob_all[:, [12]] * (sigmoid(x[:, [15]]))
        prob_all[:, [12]] = prob_all[:, [12]] * (1 - sigmoid(x[:, [15]]))

        prob_all[:, [16, 17]] = prob_all[:, [13, 14]] * (sigmoid(x[:, [16, 17]]))
        prob_all[:, [13, 14]] = prob_all[:, [13, 14]] * (1 - sigmoid(x[:, [16, 17]]))

        prob_all[:, [18, 19]] = prob_all[:, [16, 17]] * (sigmoid(x[:, [18, 19]]))
        prob_all[:, [16, 17]] = prob_all[:, [16, 17]] * (1 - sigmoid(x[:, [18, 19]]))

        if n_dim >21:
            prob_all[:, [20, 21]] = prob_all[:, [18, 19]] * (sigmoid(x[:, [20, 21]]))
            prob_all[:, [18, 19]] = prob_all[:, [18, 19]] * (1 - sigmoid(x[:, [20, 21]]))

            prob_all[:, [22, 23]] = prob_all[:, [20, 21]] * (sigmoid(x[:, [22, 23]]))
            prob_all[:, [20, 21]] = prob_all[:, [20, 21]] * (1 - sigmoid(x[:, [22, 23]]))
            
    elif model_type == 'smplh':
        prob_all = torch.ones(n_batch * n_point, n_dim-3, device=x.device)

        prob_all[:, [1, 2, 3]] = prob_all[:, [0]] * sigmoid(x[:, [0]]) * softmax(x[:, [1, 2, 3]])
        prob_all[:, [0]] = prob_all[:, [0]] * (1 - sigmoid(x[:, [0]]))

        prob_all[:, [4, 5, 6]] = prob_all[:, [1, 2, 3]] * (sigmoid(x[:, [4, 5, 6]]))
        prob_all[:, [1, 2, 3]] = prob_all[:, [1, 2, 3]] * (1 - sigmoid(x[:, [4, 5, 6]]))

        prob_all[:, [7, 8, 9]] = prob_all[:, [4, 5, 6]] * (sigmoid(x[:, [7, 8, 9]]))
        prob_all[:, [4, 5, 6]] = prob_all[:, [4, 5, 6]] * (1 - sigmoid(x[:, [7, 8, 9]]))

        prob_all[:, [10, 11]] = prob_all[:, [7, 8]] * (sigmoid(x[:, [10, 11]]))
        prob_all[:, [7, 8]] = prob_all[:, [7, 8]] * (1 - sigmoid(x[:, [10, 11]]))

        prob_all[:, [12, 13, 14]] = prob_all[:, [9]] * sigmoid(x[:, [n_dim-3]]) * softmax(x[:, [12, 13, 14]])
        prob_all[:, [9]] = prob_all[:, [9]] * (1 - sigmoid(x[:, [n_dim-3]]))
        
        prob_all[:, [15]] = prob_all[:, [12]] * (sigmoid(x[:, [15]]))
        prob_all[:, [12]] = prob_all[:, [12]] * (1 - sigmoid(x[:, [15]]))

        prob_all[:, [16, 17]] = prob_all[:, [13, 14]] * (sigmoid(x[:, [16, 17]]))
        prob_all[:, [13, 14]] = prob_all[:, [13, 14]] * (1 - sigmoid(x[:, [16, 17]]))

        prob_all[:, [18, 19]] = prob_all[:, [16, 17]] * (sigmoid(x[:, [18, 19]]))
        prob_all[:, [16, 17]] = prob_all[:, [16, 17]] * (1 - sigmoid(x[:, [18, 19]]))

        prob_all[:, [20, 21]] = prob_all[:, [18, 19]] * (sigmoid(x[:, [20, 21]]))
        prob_all[:, [18, 19]] = prob_all[:, [18, 19]] * (1 - sigmoid(x[:, [20, 21]]))

        prob_all[:, [22, 25, 28, 31, 34]] = prob_all[:, [20]] * sigmoid(x[:, [n_dim-2]]) * softmax(x[:, [22, 25, 28, 31, 34]])
        prob_all[:, [20]] = prob_all[:, [20]] * (1 - sigmoid(x[:, [n_dim-2]]))
        
        prob_all[:, [23, 26, 29, 32, 35]] = prob_all[:, [22, 25, 28, 31, 34]] * (sigmoid(x[:, [23, 26, 29, 32, 35]]))
        prob_all[:, [22, 25, 28, 31, 34]] = prob_all[:, [22, 25, 28, 31, 34]] * (1- sigmoid(x[:, [23, 26, 29, 32, 35]]))

        prob_all[:, [24, 27, 30, 33, 36]] = prob_all[:, [23, 26, 29, 32, 35]] * (sigmoid(x[:, [24, 27, 30, 33, 36]]))
        prob_all[:, [23, 26, 29, 32, 35]] = prob_all[:, [23, 26, 29, 32, 35]] * (1- sigmoid(x[:, [24, 27, 30, 33, 36]]))

        prob_all[:, [37, 40, 43, 46, 49]] = prob_all[:, [21]]  * sigmoid(x[:, [n_dim-1]]) * softmax(x[:, [37, 40, 43, 46, 49]])
        prob_all[:, [21]] = prob_all[:, [21]] * (1 - sigmoid(x[:, [n_dim-1]]))
        
        prob_all[:, [38, 41, 44, 47, 50]] = prob_all[:, [37, 40, 43, 46, 49]] * (sigmoid(x[:, [38, 41, 44, 47, 50]]))
        prob_all[:, [37, 40, 43, 46, 49]] = prob_all[:, [37, 40, 43, 46, 49]] * (1 - sigmoid(x[:, [38, 41, 44, 47, 50]]))

        prob_all[:, [39, 42, 45, 48, 51]] = prob_all[:, [38, 41, 44, 47, 50]] * (sigmoid(x[:, [39, 42, 45, 48, 51]]))
        prob_all[:, [38, 41, 44, 47, 50]] = prob_all[:, [38, 41, 44, 47, 50]] * (1 - sigmoid(x[:, [39, 42, 45, 48, 51]]))
    
    elif model_type == 'smplx':
        prob_all = torch.ones(n_batch * n_point, n_dim-4, device=x.device)

        prob_all[:, [1, 2, 3]] = prob_all[:, [0]] * sigmoid(x[:, [0]]) * softmax(x[:, [1, 2, 3]])
        prob_all[:, [0]] = prob_all[:, [0]] * (1 - sigmoid(x[:, [0]]))

        prob_all[:, [4, 5, 6]] = prob_all[:, [1, 2, 3]] * (sigmoid(x[:, [4, 5, 6]]))
        prob_all[:, [1, 2, 3]] = prob_all[:, [1, 2, 3]] * (1 - sigmoid(x[:, [4, 5, 6]]))

        prob_all[:, [7, 8, 9]] = prob_all[:, [4, 5, 6]] * (sigmoid(x[:, [7, 8, 9]]))
        prob_all[:, [4, 5, 6]] = prob_all[:, [4, 5, 6]] * (1 - sigmoid(x[:, [7, 8, 9]]))

        prob_all[:, [10, 11]] = prob_all[:, [7, 8]] * (sigmoid(x[:, [10, 11]]))
        prob_all[:, [7, 8]] = prob_all[:, [7, 8]] * (1 - sigmoid(x[:, [10, 11]]))

        prob_all[:, [12, 13, 14]] = prob_all[:, [9]] * sigmoid(x[:, [n_dim-4]]) * softmax(x[:, [12, 13, 14]])
        prob_all[:, [9]] = prob_all[:, [9]] * (1 - sigmoid(x[:, [n_dim-4]]))

        prob_all[:, [15]] = prob_all[:, [12]] * (sigmoid(x[:, [15]]))
        prob_all[:, [12]] = prob_all[:, [12]] * (1 - sigmoid(x[:, [15]]))

        prob_all[:, [16, 17]] = prob_all[:, [13, 14]] * (sigmoid(x[:, [16, 17]]))
        prob_all[:, [13, 14]] = prob_all[:, [13, 14]] * (1 - sigmoid(x[:, [16, 17]]))

        prob_all[:, [18, 19]] = prob_all[:, [16, 17]] * (sigmoid(x[:, [18, 19]]))
        prob_all[:, [16, 17]] = prob_all[:, [16, 17]] * (1 - sigmoid(x[:, [18, 19]]))

        prob_all[:, [20, 21]] = prob_all[:, [18, 19]] * (sigmoid(x[:, [20, 21]]))
        prob_all[:, [18, 19]] = prob_all[:, [18, 19]] * (1 - sigmoid(x[:, [20, 21]]))

        prob_all[:, [22, 23, 24]] = prob_all[:, [15]] * sigmoid(x[:, [n_dim-3]]) * softmax(x[:, [22, 23, 24]])
        prob_all[:, [15]] = prob_all[:, [15]] * (1 - sigmoid(x[:, [n_dim-3]]))

        prob_all[:, [25, 28, 31, 34, 37]] = prob_all[:, [20]] * sigmoid(x[:, [n_dim-2]]) * softmax(x[:, [25, 28, 31, 34, 37]])
        prob_all[:, [20]] = prob_all[:, [20]] * (1 - sigmoid(x[:, [n_dim-2]]))

        prob_all[:, [26, 29, 32, 35, 38]] = prob_all[:, [25, 28, 31, 34, 37]] * (sigmoid(x[:, [26, 29, 32, 35, 38]]))
        prob_all[:, [25, 28, 31, 34, 37]] = prob_all[:, [25, 28, 31, 34, 37]] * (1- sigmoid(x[:, [26, 29, 32, 35, 38]]))

        prob_all[:, [27, 30, 33, 36, 39]] = prob_all[:, [26, 29, 32, 35, 38]] * (sigmoid(x[:, [27, 30, 33, 36, 39]]))
        prob_all[:, [26, 29, 32, 35, 38]] = prob_all[:, [26, 29, 32, 35, 38]] * (1- sigmoid(x[:, [27, 30, 33, 36, 39]]))

        prob_all[:, [40, 43, 46, 49, 52]] = prob_all[:, [21]]  * sigmoid(x[:, [n_dim-1]]) * softmax(x[:, [40, 43, 46, 49, 52]])
        prob_all[:, [21]] = prob_all[:, [21]] * (1 - sigmoid(x[:, [n_dim-1]]))
        
        prob_all[:, [41, 44, 47, 50, 53]] = prob_all[:, [40, 43, 46, 49, 52]] * (sigmoid(x[:, [41, 44, 47, 50, 53]]))
        prob_all[:, [40, 43, 46, 49, 52]] = prob_all[:, [40, 43, 46, 49, 52]] * (1 - sigmoid(x[:, [41, 44, 47, 50, 53]]))

        prob_all[:, [42, 45, 48, 51, 54]] = prob_all[:, [41, 44, 47, 50, 53]] * (sigmoid(x[:, [42, 45, 48, 51, 54]]))
        prob_all[:, [41, 44, 47, 50, 53]] = prob_all[:, [41, 44, 47, 50, 53]] * (1 - sigmoid(x[:, [42, 45, 48, 51, 54]]))
    
    elif model_type == 'mano':
        prob_all = torch.ones(n_batch * n_point, n_dim-1, device=x.device)
        prob_all[:, [1, 4, 7, 10, 13]] = prob_all[:, [0]] * sigmoid(x[:, [0]]) * softmax(x[:, [1, 4, 7, 10, 13]])
        prob_all[:, [0]] = prob_all[:, [0]] * (1 - sigmoid(x[:, [0]]))

        prob_all[:, [2, 5, 8, 11, 14]] = prob_all[:, [1, 4, 7, 10, 13]] * (sigmoid(x[:, [2, 5, 8, 11, 14]]))
        prob_all[:, [1, 4, 7, 10, 13]] = prob_all[:, [1, 4, 7, 10, 13]] * (1 - sigmoid(x[:, [2, 5, 8, 11, 14]]))

        prob_all[:, [3, 6, 9, 12, 15]] = prob_all[:, [2, 5, 8, 11, 14]] * (sigmoid(x[:, [3, 6, 9, 12, 15]]))
        prob_all[:, [2, 5, 8, 11, 14]] = prob_all[:, [2, 5, 8, 11, 14]] * (1 - sigmoid(x[:, [3, 6, 9, 12, 15]]))
    
    elif model_type == 'flame':
        prob_all = torch.ones(n_batch * n_point, n_dim-1, device=x.device)
        prob_all[:, [1, 2, 3]] = prob_all[:, [0]] * sigmoid(x[:, [0]]) * softmax(x[:, [1, 2, 3]])
        prob_all[:, [0]] = prob_all[:, [0]] * (1 - sigmoid(x[:, [0]]))
    
    else:
        raise NotImplementedError
    
    prob_all = prob_all.reshape(n_batch, n_point, prob_all.shape[-1])
    
    return prob_all