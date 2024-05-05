import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse 
from time import time

from utils.dataprep import matrix_from_observations, transform_indices, reindex_data
from utils.evaluation import model_evaluate
from utils.dataset import Dataset

from cdimf import CDIMF

def create_arg_parser():
    """Create argument parser for our baseline. """
    parser = argparse.ArgumentParser('RS24')

    # DATA  Arguments
    parser.add_argument('--domains', type=str, default="sport_cloth || electronic_cell, sport_cloth || game_video")
    parser.add_argument('--task', type=str, default='warm-start', help='cold-start or warm-start')

    # MODEL Arguments
    parser.add_argument('--model', type=str, default='CDIMF', help='CDIMF, ALS_separate or ALS_joined')
    parser.add_argument('--num_epoch', type=int, help='number of epoches')
    parser.add_argument('--prox', type=str, help="proximal operator for factors' aggregation")
    parser.add_argument('--reg_z', type=float, help='the L2 weight for proximal operator')
    parser.add_argument('--reg_als', type=float, help='regularization for the ALS solver')
    parser.add_argument('--unobserved_weight', type=float, help='unobserved weight')
    parser.add_argument('--rho', type=float, help='cross-domain sharing parameter')
    parser.add_argument('--v', type=float, help='frequency-scaled regularization parameter')
    parser.add_argument('--rank', type=int, help='latent dimension')

    # others
    parser.add_argument('--seed', type=int, default=15, help='manual seed init')

    return parser
   
def run_cdimf_experiment(nodes, 
                         datasets, 
                         n_epochs=1,
                         evaluate_every=0,
                         aggregate_every=1,
                         topn=10,
                         verbose=0, 
                         inter_domain=False):
    metrics = []
    for _ in range(len(datasets)):
        metrics.append(dict())

    # report initial metrics
    if evaluate_every > 0:
        start_time = time()
        for i in range(len(datasets)):
            ds = datasets[i]
            node = nodes[i] if len(nodes)>1 else nodes[0]
            recs = node.get_recommendations(ds.training, samples=ds.samples, topn=topn, 
                                            seen_items_excluded=ds.seen_items_excluded,
                                            from_foreign_domain=nodes[1-i] if inter_domain else None)
            metrics_ = model_evaluate(recs, ds.test, ds.description, sampling_method=ds.sampling_method) 
            metrics[i]['epochs'] = [0]
            for name, value in metrics_.items():
                metrics[i][name] = [value]
            if verbose>0:
                print(f'node {i+1 if ds.name is None else ds.name} metrics : {metrics_}')
        end_time = time()
        if verbose>0: print('Evaluation time = ', end_time-start_time, 's')
        

    for epoch in range(n_epochs):
        start_time = time()
        if verbose>0: print(f'Epoch {epoch+1}')
        XU = []
        # train locally
        for node in nodes:
            XU.append(node.train(iterations=1, verbose=verbose))
        # aggregate
        if epoch % aggregate_every == 0:
            for i, node in enumerate(nodes):
                if verbose>0:
                    print(f'\rAggregate nodes {i+1}/{len(nodes)}', end='')
                node.aggregate(XU)
            if verbose>0: print()
        end_time = time()
        if verbose>0: print('Traing time = ', end_time-start_time, 's')
        
        # evaluate
        if evaluate_every>0 and (epoch+1)%evaluate_every==0:
            start_time = time()
            for i in range(len(datasets)):
                ds = datasets[i]
                node = nodes[i] if len(nodes)>1 else nodes[0]
                recs = node.get_recommendations(ds.training, samples=ds.samples, topn=topn, 
                                                seen_items_excluded=ds.seen_items_excluded,
                                                from_foreign_domain=nodes[1-i] if inter_domain else None)
                metrics_ = model_evaluate(recs, ds.test, ds.description, sampling_method=ds.sampling_method) 
                metrics[i]['epochs'].append(metrics[i]['epochs'][-1]+evaluate_every)
                for name, value in metrics_.items():
                    metrics[i][name].append(value)
                if verbose>0:
                    print(f'node {i+1 if ds.name is None else ds.name} metrics : {metrics_}')
            end_time = time()
            if verbose>0: print('Evaluation time = ', end_time-start_time, 's')

    return metrics

def load_params(opt, params_file):
    params_df = pd.read_csv(params_file)
    defaults = params_df[np.all([
                params_df['task']==opt['task'], 
                params_df['model']==opt['model'], 
                params_df['domains']==opt['domains']
                ], axis=0)].to_dict(orient='index')
    if len(defaults)==0:
        print('No default parameters for your experiments!')
        params = {}
    else:
        params = next(iter(defaults.values()))
    for k, v in opt.items():
        if not v is None:
            params[k] = v 
    print(params)
    return params

if __name__=="__main__":
    parser = create_arg_parser()
    args = parser.parse_args()
    opt = vars(args)

    domains = opt['domains']
    task = opt['task']
    N1, N2 = domains.split('_')
    dataset1_name = N1 +'_'+ N2
    dataset2_name = N2 +'_'+ N1
    opt['domains'] = min(dataset1_name, dataset2_name)

    data_description = dict(
        users = 'userid',
        items = 'itemid', 
        feedback = 'rating' 
    )

    domain1 = pd.read_csv(f'datasets/{task}/dataset/{dataset1_name}/train.txt', header=None, names=['userid', 'itemid', 'rating'], sep='\t', )
    domain2 = pd.read_csv(f'datasets/{task}/dataset/{dataset2_name}/train.txt', header=None, names=['userid', 'itemid', 'rating'], sep='\t')
    domain1_test = pd.read_csv(f'datasets/{task}/dataset/{dataset1_name}/test.txt', header=None, names=['userid', 'itemid', 'rating'], sep='\t')
    domain2_test = pd.read_csv(f'datasets/{task}/dataset/{dataset2_name}/test.txt', header=None, names=['userid', 'itemid', 'rating'], sep='\t')

    for df in domain1, domain1_test,domain2, domain2_test:
        df.fillna(1, inplace=True)

    data_description1 = data_description.copy()
    data_description1['n_items'] = domain1[data_description['items']].nunique()
    data_description1['n_users'] = domain1[data_description['users']].nunique()
    data_description2 = data_description.copy()
    data_description2['n_items'] = domain2[data_description['items']].nunique()
    data_description2['n_users'] = domain2[data_description['users']].nunique()

    if opt['model'] == 'ALS_joined': 
        # add prefix to itemid's before mixing the dataframes
        domain1['itemid'] = domain1['itemid'].apply(lambda x:'A'+str(x))
        domain2['itemid'] = domain2['itemid'].apply(lambda x:'B'+str(x))
        domain1_test['itemid'] = domain1_test['itemid'].apply(lambda x:'A'+str(x))
        domain2_test['itemid'] = domain2_test['itemid'].apply(lambda x:'B'+str(x))

        # mix the training dataframes and index each domain's dataframe by  the mixed index
        mixed_domains = pd.concat([domain1, domain2], axis=0)
        training ,mixed_index = transform_indices(mixed_domains, 'userid', 'itemid')

        data_description = data_description.copy()
        data_description['n_items'] = training[data_description['items']].nunique()
        data_description['n_users'] = training[data_description['users']].nunique()

        mtx = matrix_from_observations(training, data_description)

        training1 = reindex_data(domain1, data_index=mixed_index)
        training2 = reindex_data(domain2, data_index=mixed_index)

        test1 = reindex_data(domain1_test, data_index=mixed_index)
        test2 = reindex_data(domain2_test, data_index=mixed_index)
    
    else: # the cases of model=='CDIMF' or 'ALS_separate' are the same for indexing
        training1 ,domain1_index = transform_indices(domain1, 'userid', 'itemid')
        training2 ,domain2_index = transform_indices(domain2, 'userid', 'itemid')

        mtx1 = matrix_from_observations(training1, data_description1)
        mtx2 = matrix_from_observations(training2, data_description2)

        if opt['task']=='warm-start':
            test1 = reindex_data(domain1_test, data_index=domain1_index)
            test2 = reindex_data(domain2_test, data_index=domain2_index)

        else: # index testsets by target domain index for users and by source domain index for items 
            test1 = reindex_data(
                reindex_data(domain1_test, 
                            data_index=domain1_index, fields='items'),
                            data_index=domain2_index, fields='users')
            test2 = reindex_data(
                reindex_data(domain2_test,
                            data_index=domain2_index, fields='items'),
                            data_index=domain1_index, fields='users')
    
        domain1_users = domain1['userid'].unique()
        domain2_users = domain2['userid'].unique()
        shared_users = np.intersect1d(domain2_users, domain1_users)
        print('domain1 users:', len(domain1_users), 
            'domain2 users:',len(domain2_users), 
            'shared users:',len(shared_users))
        
        # get the indiced for shared users in each domain 
        common_users_1 = domain1_index['users'].get_indexer(shared_users)
        common_users_2 = domain2_index['users'].get_indexer(shared_users)

    # build datasets and generate samples

    print('generate test samples ...')
    ds1 = Dataset(training=training1, test=test1, description=data_description1, name=N1)
    ds2 = Dataset(training=training2, test=test2, description=data_description2, name=N2)
    for ds in [ds1, ds2]:
        ds.generate_samples(negatives_per_item=999, 
                            sampling_method='per_item', 
                            exclude_seen_items=False, 
                            item_catalog='relative')
        

    params = load_params(opt, params_file='default_params.csv')

    if opt['model'] != 'ALS_joined':
        node1 = CDIMF(data=mtx1, common_users=common_users_1, params=params)
        node2 = CDIMF(data=mtx2, common_users=common_users_2, params=params)
        nodes = [node1, node2]
    else:
        nodes = [CDIMF(data=mtx, common_users=data_description['n_users'], params=params)]

    
    np.random.seed(opt['seed'])
    metrics = run_cdimf_experiment(nodes=nodes,
                                   datasets=[ds1, ds2],
                                   n_epochs=params['num_epoch'],
                                   evaluate_every=1,
                                   aggregate_every=1,
                                   verbose=1,
                                   inter_domain=(params['task']=='cold-start' and params['model']!='ALS_joined'))


    













        