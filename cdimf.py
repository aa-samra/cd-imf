import numpy as np
from tqdm import tqdm
import scipy
from scipy.linalg import sqrtm
from utils.evaluation import topn_recommendations, model_evaluate
from joblib import Parallel, delayed
from time import time

import os
os.environ["TQDM_DISABLE"] = "1"

def notqdm(it, *a, **k):
    return it

# tqdm = nop

class CDIMF:
    def __init__(self, data, common_users=None, params={}, njobs=-1, name=None) -> None:
        self.data = scipy.sparse.csr_matrix(data, dtype=np.float32)
        self.cdata = self.data.tocsc()
        assert len(data.shape) == 2
        self.nU, self.nI = self.data.shape
        self.njobs = njobs
        self.name = name

        # Common users handleing: None->All, n:int->the first n users, iteratable->specific indices 
        if common_users is None:
            self.common_users = np.arange(common_users)
            self.non_common_users = []
        elif isinstance(common_users, int):
            self.common_users = np.arange(common_users)
            self.non_common_users = np.arange(common_users, self.nU)
        else:
            try:
                _ = iter(common_users)
                self.common_users = np.array(common_users)
                self.non_common_users = np.setdiff1d(np.arange(self.nU), self.common_users)
            except TypeError as te:
                print(common_users, 'is not iterable') 

        # Initialization hyperparameters
        self.rank = params.get('rank', 50)
        self.std = params.get('std', 0.1)
        self.seed = params.get('seed', 15)
        # ALS hyperparameters
        self.implicit = params.get('implicit', True)
        self.reg_als = params.get('reg_als', 0.1)
        self.unobserved_weight = params.get('unobserved_weight', 0.3)
        self.v = params.get('v', 0)
        
        # ADMM aggregation hyperparameters
        self.reg_z = params.get('reg_z', 0.1)
        self.rho = params.get('rho', 1)  # control cross-domain sharing (set to 0 for local ALS, other for standard ADMM)
        self.prox = params.get('prox', lambda x:x) # identity prox

        np.random.seed(self.seed)
        self.X = np.float32(self.std * np.random.randn(self.nU, self.rank) / np.sqrt(self.rank))
        self.Y = np.float32(self.std * np.random.randn(self.nI, self.rank) / np.sqrt(self.rank))

        self.Z = np.zeros((len(self.common_users), self.rank), dtype=np.float32)
        self.U = np.zeros_like(self.Z, dtype=np.float32)
        self.mixed_XU = np.zeros_like(self.Z, dtype=np.float32)
    
    @staticmethod
    # @jit(nopython=True)
    def solve_one_row_col(i ,data, idx, ptr, E, alpha,
                          full_similarity_mtx, global_reg,
                          source_factors, local_reg,
                          additional_term=None):

        X = source_factors[idx[ptr[i]:ptr[i+1]]]
        p = data[ptr[i]:ptr[i+1]]

        Ai = alpha * full_similarity_mtx + X.T @ X + global_reg * local_reg[i]**2 * E 
        Vi = X.T @ p

        if not additional_term is None:
            assert Vi.shape == additional_term.shape
            Vi += additional_term * global_reg * local_reg[i]

        target_factor = np.linalg.solve(Ai, Vi)
        return target_factor

    def train(self, iterations=1, verbose=0):
        P_csr = self.data
        P_csc = self.cdata

        rdata, ridx, rptr= P_csr.data, P_csr.indices, P_csr.indptr
        Rx = np.float32((np.diff(rptr) / self.nI + self.unobserved_weight) ** self.v)

        cdata, cidx, cptr= P_csr.data, P_csc.indices, P_csc.indptr
        Ry = np.float32((np.diff(cptr) / self.nU + self.unobserved_weight) ** self.v)

        E = np.eye(self.rank , dtype=np.float32)
        
        if self.implicit:
            rdata = np.ones_like(rdata, dtype=np.float32)
            cdata = np.ones_like(cdata, dtype=np.float32)

        for iteration in range(iterations):
            # Update users
            phase = 'users ...'
            if verbose==1:
                print(f"\rIteration: {(iteration+1):d}/{iterations} updating {phase}", end="")
            YtY = self.Y.T @ self.Y
            # common users
            self.X[self.common_users] = np.vstack(Parallel(n_jobs=self.njobs)(
                delayed(self.solve_one_row_col) (i=i,
                                                 data=rdata, 
                                                 idx=ridx, 
                                                 ptr=rptr, 
                                                 E=E, 
                                                 alpha=self.unobserved_weight,
                                                 full_similarity_mtx=YtY,
                                                 global_reg=self.reg_als,
                                                 source_factors=self.Y, 
                                                 local_reg=Rx,
                                                 additional_term=self.rho*(self.Z[k]-self.U[k])) 
                for k,i in enumerate(self.common_users)))
            # non common users
            if len(self.non_common_users)>0:
                self.X[self.non_common_users] = np.vstack(Parallel(n_jobs=self.njobs)(
                    delayed(self.solve_one_row_col) (i=i, 
                                                     data=rdata, 
                                                     idx=ridx, 
                                                     ptr=rptr, 
                                                     E=E, 
                                                     alpha=self.unobserved_weight,
                                                     full_similarity_mtx=YtY, 
                                                     global_reg=self.reg_als,
                                                     source_factors=self.Y, 
                                                     local_reg=Rx,
                                                     additional_term=None) 
                    for i in self.non_common_users))

            phase = 'users done, items ...'
            if verbose==1:
                print(f"\rIteration: {(iteration+1):d}/{iterations} updating {phase}", end="")
            # update items
            XtX = self.X.T @ self.X
            self.Y= np.vstack(Parallel(n_jobs=self.njobs)(
                    delayed(self.solve_one_row_col) (i=j, 
                                                     data=cdata, 
                                                     idx=cidx, 
                                                     ptr=cptr, 
                                                     E=E, 
                                                     alpha=self.unobserved_weight, 
                                                     full_similarity_mtx=XtX, 
                                                     global_reg=self.reg_als,
                                                     source_factors=self.X, 
                                                     local_reg=Ry,
                                                     additional_term=None) 
                    for j in range(self.nI)))
            
        if self.prox == 'ortho':
            Q, R = np.linalg.qr(self.X)
            self.X = Q
            self.Y = self.Y @ R.T

        self.mixed_XU = (self.X[self.common_users] + self.U)
        
        phase = 'users done, items done '
        if verbose==1:
            print(f"\rIteration: {(iteration+1):d}/{iterations} updating {phase}", end="")
            print()

        return self.mixed_XU

    def aggregate(self, xu, self_param_included=True):
        if self_param_included:
            N_sides = len(xu)
            mean_xu = np.sum(xu, axis=0) / N_sides
        else:
            N_sides = len(xu) + 1
            mean_xu = (self.mixed_XU + np.sum(xu, axis=0)) / N_sides

        if self.prox is None:
            self.Z = mean_xu
        if self.prox == 'L2':
            self.Z = N_sides * self.rho * mean_xu / (N_sides * self.rho + self.reg_z + 1e-15) 
        if self.prox == 'ortho':
            self.Z = mean_xu @ sqrtm(mean_xu.T @ mean_xu)

        self.U = self.U + (self.X[self.common_users] - self.Z)
        
        return self.Z
    
    def get_recommendations(self, training, topn=10, samples={}, from_foreign_domain=None, seen_items_excluded=False):
        def rec_for_user(userid, sampled_items):
            if from_foreign_domain is None:
                user_factor = self.X[userid]
            else:
                user_factor = from_foreign_domain.X[userid]
            scores = user_factor @ self.Y[sampled_items].T
            #downvote_seen_items 
            if not seen_items_excluded:
                seen_items = training.query(f'userid == @userid')['itemid'].values
                seen_idx = np.where(np.isin( sampled_items, seen_items))
                scores[seen_idx] = scores.min()-1
            recs_idx = topn_recommendations(scores.reshape(1,-1), topn=topn)
            return sampled_items[recs_idx]

        recs = np.vstack(Parallel(n_jobs=self.njobs)(
                delayed(rec_for_user) (userid, sampled_items) 
                for userid, sampled_items in samples))
        return recs
