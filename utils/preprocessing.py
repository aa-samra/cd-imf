import pandas as pd
import numpy  as np

def pq_core_filter(data, data_description, min_activity=1, min_popularity=1, single_iteration=False):
    """ 
    increase the density of data by dropping least active users and least popular items
    data: pandas.dataframe
    data_description: a dict contains the col name of userid and itemid
    min_activity: the minimum number of user's interactions in the output dataframe  
    min_popularity: the minimum number of item's interactions in the output dataframe  
    """
    itemid = data_description['items']
    userid = data_description['users']
    dense_data = data.copy()
    n=0
    finish = False
    a, b = 1, 1
    nU, nI = dense_data[userid].nunique(), dense_data[itemid].nunique()
    d = len(dense_data) / (nU*nI) 
    print('#iter', 'interactions', 'users', 'items' , 'min_activity', 'min_popularity', 'density', sep='\t\t')
    print(n, len(dense_data), nU, nI, a, b, d, sep='\t\t')
    while len(dense_data)>0 and not finish:
        active_users = dense_data.groupby(userid).size() >= min_activity
        active_users = active_users.index[active_users]
        
        popular_items = dense_data.groupby(itemid).size() >= min_popularity
        popular_items = popular_items.index[popular_items]

        dense_data = dense_data.query(f'{userid} in @active_users')
        dense_data = dense_data.query(f'{itemid} in @popular_items')
        
        n += 1
        a = dense_data.groupby('userid').size().min()
        b = dense_data.groupby('itemid').size().min()
        nU, nI = dense_data[userid].nunique(), dense_data[itemid].nunique()
        d = len(dense_data) / (nU*nI)  
        print(n, len(dense_data), nU, nI, a, b, d, sep='\t\t')
        
        finish = a>=min_activity and b>=min_popularity
        if single_iteration: break
    return dense_data


def index_cd_data(data1, data2, data_description, common_size=None, filter_items=0):
    '''
    index 2 dataframes consistently so shared user have indices [0, n_common_users]
    and non shared will take next indices. 
    data1, data2: pd.DataFrame contains user-item interaction
    data_description: dict() with names of users/items columns
    common_size: ratio ]0.0-1.0] control the ratio of common to non-common parts
    RETURNS: ata1, data2, data_index1, data_index2, n_common_users

    '''
    userid = data_description['users']
    itemid = data_description['items']
    users1 = data1[userid].unique()
    users2 = data2[userid].unique()
    common_users = np.intersect1d(users1, users2)

    d1_common_first_idx = pd.Index(common_users)
    d2_common_first_idx = pd.Index(common_users)

    if not common_size is None and common_size<1:
        d1_only_users = np.setdiff1d(users1, users2)
        d2_only_users = np.setdiff1d(users2, users1)
        print(f"d1_only: {len(d1_only_users)}, d2_only: {len(d2_only_users)}")
        assert common_size>0 and common_size<=1.0, "ratio should be in ]0,1]"
        r = (1-common_size) / common_size # non-common to common ratio
        d1_only_size = min(len(d1_only_users), int(len(common_users) * r))
        d2_only_size = min(len(d2_only_users), int(len(common_users) * r))

        d1_only_users = np.random.choice(d1_only_users, d1_only_size, replace=False)
        d2_only_users = np.random.choice(d2_only_users, d2_only_size, replace=False)

        d1_common_first_idx = pd.Index(np.concatenate([common_users, d1_only_users]))
        d2_common_first_idx = pd.Index(np.concatenate([common_users, d2_only_users]))
        
    
    print(f"common: {len(common_users)}")

    #reindex users
    data1 = data1.assign(**{userid: d1_common_first_idx.get_indexer(data1[userid])}) 
    data2 = data2.assign(**{userid: d2_common_first_idx.get_indexer(data2[userid])}) 

    data1.drop(data1.query(f'{userid} < 0').index, inplace=True)
    data2.drop(data2.query(f'{userid} < 0').index, inplace=True)

    #reindex items
    if not common_size is None and filter_items>0:
        data1 = pq_core_filter(data1, data_description, min_activity=1, min_popularity=filter_items)
        data2 = pq_core_filter(data2, data_description, min_activity=1, min_popularity=filter_items)

    d1_items_idx = pd.Index(data1[itemid].unique())
    d2_items_idx = pd.Index(data2[itemid].unique())
    data1 = data1.assign(**{itemid: d1_items_idx.get_indexer(data1[itemid])}) 
    data2 = data2.assign(**{itemid: d2_items_idx.get_indexer(data2[itemid])}) 


    
    data_index1 = {
        'users': d1_common_first_idx,
        'items': d1_items_idx
    }
    data_index2 = {
        'users': d2_common_first_idx,
        'items': d2_items_idx
    }

    return data1, data2, data_index1, data_index2, len(common_users)

def split_train_test_inter(data1, data2, data_description, test_size, seed=15):
    userid = data_description['users']
    common_users = np.intersect1d(data1[userid].unique(), data2[userid].unique())
    n_common_users = len(common_users)
    np.random.RandomState(seed=seed)
    test_users = np.random.choice(common_users, size=int(n_common_users*test_size), replace=True)
    n_test_users = len(test_users)
    test_users_1 = test_users[:n_test_users//2]
    test_users_2 = test_users[n_test_users//2:]
    print('common_users', n_common_users, '  test1_users', len(test_users_1), '  test2_users', len(test_users_2))

    test1 = data1.query(f"{userid} in @test_users_1")
    test2 = data2.query(f"{userid} in @test_users_2")

    train1 = data1.drop(test1.index, axis=0)
    train2 = data2.drop(test2.index, axis=0)

    return train1, test1, train2, test2
