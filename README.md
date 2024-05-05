# CDIMF: Cross-Domain Implicit Matrix Factorization
## Overview
This repository contains the code for the CDIMF (Cross-Domain Implicit Matrix Factorization) model, as described in the research paper (abstact below). CDIMF is a matrix factorization-based approach for cross-domain recommender systems that aims to address the data sparsity problem by leveraging knowledge from other source domains.

## Abstract
The abstract of the research paper is as follows:
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
* `--prox`: Specifies the proximal operator for factors' aggregation.
* `--reg_z`: Specifies the L2 weight for the proximal operator.
* `--reg_als`: Specifies the regularization for the ALS solver.
* `--unobserved_weight`: Specifies the unobserved weight.
* `--rho`: Specifies the cross-domain sharing parameter.
* `--v`: Specifies the frequency-scaled regularization parameter.
* `--rank`: Specifies the latent dimension.
* `--seed`: Specifies the manual seed initialization.

## Requirements
* Python 3.11
* NumPy
* SciPy
* Pandas
* Joblib
* tqdm
