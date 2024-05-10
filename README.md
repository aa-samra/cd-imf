# CDIMF: Cross-Domain Implicit Matrix Factorization
## Overview
This repository contains the code for the CDIMF (Cross-Domain Implicit Matrix Factorization) model, as described in the research paper (abstact below). CDIMF is a matrix factorization-based approach for cross-domain recommender systems that aims to address the data sparsity problem by leveraging knowledge from other source domains.

## Abstract
The abstract of the corresponding research paper is as follows:
> Data sparsity has been one of the long-standing problems for recommender systems. One of the solutions to mitigate this issue is to exploit knowledge available in other source domains. However, many cross-domain recommender systems introduce a complex architecture that makes them less scalable in practice. On the other hand, matrix factorization methods are still considered to be strong baselines for single-domain recommendations. In this paper, we introduce the CDIMF, a model that extends the standard implicit matrix factorization with ALS to cross-domain scenarios. We apply the Alternating Direction Method of Multipliers to learn shared latent factors for overlapped users while factorizing the interaction matrix. In a dual-domain setting, experiments on industrial datasets demonstrate a competing performance of CDIMF for both cold-start and warm-start. The proposed model can outperform most other recent cross-domain and single-domain models.

## Usage
To run experiments with the CDIMF model, you can use the provided argument parser. Here's an example command:

``` $ python run_cdimf.py --domains sport_cloth --task warm-start --model CDIMF ```

The available arguments are:
* `--model`: Specifies the model to use, either "CDIMF", "ALS_separate", or "ALS_joined".
* `--task`: Specifies the task, either "cold-start" or "warm-start".
* `--domains`: Specifies the source and target domains, e.g., "sport_cloth, electronic_phone" for warm-start, and "sport_cloth,  game_video" for cold-start.

These parameters are optional. If not specified, the default values for the corresponding experiment will be used: 
* `--num_epoch`: Specifies the number of epochs.
* `--rank`: Specifies the latent dimension.
* `--reg_als`: Specifies the regularization for the ALS solver.
* `--unobserved_weight`: Specifies the unobserved weight.
* `--v`: Specifies the frequency-scaled regularization parameter.
* `--rho`: Specifies the cross-domain sharing parameter.
* `--prox`: Specifies the proximal operator for factors' aggregation, either "L2" or "identity". 
* `--reg_z`: Specifies the L2 regularization parameter if the proximal operator is "L2".
* `--seed`: Specifies the manual seed initialization.
* `--evaluate_every`: Set evaluation frequency
* `--aggregate_every`: Set shared parameters aggregation frequency
* `--verbose`: Set output verbosity, 0-silent, 1-meduim, 2-verbose'


## Requirements
* Python 3.11
* NumPy
* SciPy
* Pandas
* Joblib
* tqdm

## Model Analysis
CDIMF has three characteristic hyperparameters: the sharing (penalty) parameter, the proximal operator, and the aggregation period. We conducted several experiments to understand the effect of each hyperparameter on the behavior of CDIMF. Results of these expeiments can be found in `model_analysis.ipynb`

## How to run the model it for your data 
These steps help you to run the CDIMF on your own data. Currently, they aim to reproduce results for research only. More instruction about how to deploy the `CDIMF` class in your project will be added later.   
1. Split the data of each domain into `train.txt` and `test.txt`.
2. Put the files of the first domain in a directory named `D1_D2` and the second domain in a directory named `D2_D1`
3. Place the both directorys in either `datasets/cold-start` or `datasets/warm-start` depending on the experiment task.
4. Run this code.
```python
from run_cdimf import run_cdimf_experiment
args = {
    'domains': "D1_D2",
    'task': 'warm-start',
    'model': 'CDIMF',
    'num_epoch': 12,
    'prox': 'L2',
    'reg_z': 0,
    'reg_als': 0.03,
    'unobserved_weight': 0.3,
    'rho': 0,
    'v': 0.5,
    'rank': 192,
    'seed': 15,
    'evaluate_every': 1,
    'aggregate_every': 1,
    'verbose': 0
}

metrics = run_cdimf_experiment(args)
```
The output `metrics` has the following structure:
```py
metrics = [
    { # the first domain  
        'epochs': [0, ...], # the epochs when the model is evaluated, 0 for inital metrics- random
        'HR@10' : [...], # Hit rates
        'NDCG@10': [...], # Normalized Decreasing Cumulative Gain
        'COV@10': [...] # Covarage rate
    },
    { # the second domain
        'epochs': [0, ...],
        'HR@10' : [...],
        'NDCG@10': [...],
        'COV@10': [...]
    }
]
```