import numpy as np
import pandas as pd


def downvote_negative_samples(scores, holdout, data_description, negative_samples=999):
    assert isinstance(scores, np.ndarray), 'Scores must be a dense numpy array!'
    itemid = data_description['items']
    userid = data_description['users']
    for row, true_item in enumerate(holdout[itemid].values):
        true_item_score = scores[row, true_item]
        drop_indices = np.random.choice(data_description['n_items'], 
                                        size=data_description['n_items'] - negative_samples - 1,
                                        replace=False)
        scores[row, drop_indices] = scores.min() - 1
        scores[row, true_item] = true_item_score

def downvote_seen_items(scores, data, data_description):
    assert isinstance(scores, np.ndarray), 'Scores must be a dense numpy array!'
    itemid = data_description['items']
    userid = data_description['users']
    # get indices of observed data, corresponding to scores array
    # we need to provide correct mapping of rows in scores array into
    # the corresponding user index (which is assumed to be sorted)
    row_idx, test_users = pd.factorize(data[userid], sort=True)
    assert len(test_users) == scores.shape[0]
    col_idx = data[itemid].values
    # downvote scores at the corresponding positions
    scores[row_idx, col_idx] = scores.min() - 1


def topn_recommendations(scores, topn=10):
    recommendations = np.apply_along_axis(topidx, 1, scores, topn)
    return recommendations


def topidx(a, topn):
    parted = np.argpartition(a, -topn)[-topn:]
    return parted[np.argsort(-a[parted])]


def model_evaluate(recommended_items, holdout, holdout_description, topn=10, sampling_method='per_user'):
    itemid = holdout_description['items']
    userid = holdout_description['users']
    n_test_users = recommended_items.shape[0]

    if sampling_method=='per_user':
        hits_mask = np.empty_like(recommended_items[:, :topn], dtype=np.int8)
        for i, positive_items in enumerate(holdout.groupby(userid)[itemid].agg(list)):
            hits_mask[i] = np.isin(recommended_items[i, :topn], positive_items) 

        # find the rank of the EARLIEST true item 
        hit_rank = np.argmax(hits_mask, axis=1) + (hits_mask.sum(axis=1)>0).astype(int)
        # keep nozero ranks only 
        hit_rank = hit_rank[np.nonzero(hit_rank)]
    elif sampling_method=='per_item':
        holdout_items = holdout[itemid].values
        assert recommended_items.shape[0] == len(holdout_items)
        hits_mask = recommended_items[:, :topn] == holdout_items.reshape(-1, 1)
        hit_rank = np.where(hits_mask)[1] + 1.0
    
    # HR calculation
    hr = len(hit_rank) / n_test_users
    # MRR calculation
    mrr = np.sum(1 / hit_rank) / n_test_users
    # coverage calculation
    n_items = holdout_description['n_items']
    cov = np.unique(recommended_items).size / n_items
        # NDCG
    ndcg_pu = 1.0 / np.log2(hit_rank + 1)
    ndcg = np.sum(ndcg_pu) / n_test_users

    return {f'HR@{topn}': hr, f'NDCG@{topn}':ndcg, f'MRR@{topn}':mrr, f'COV@{topn}':cov}


def calculate_rmse(scores, holdout, holdout_description):
    user_idx = np.arange(holdout.shape[0])
    item_idx = holdout[holdout_description['items']].values
    feedback = holdout[holdout_description['feedback']].values
    predicted_rating = scores[user_idx, item_idx]
    return np.mean(np.abs(predicted_rating-feedback)**2)