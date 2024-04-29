from scipy.sparse import csr_matrix
import pandas as pd
import numpy as np

def leave_last_out(data, userid='userid', timeid='timestamp'):
    data_sorted = data.sort_values('timestamp')
    holdout = data_sorted.drop_duplicates(
        subset=['userid'], keep='last'
    ) # split the last item from each user's history
    remaining = data.drop(holdout.index) # store the remaining data - will be our training
    return remaining, holdout

def leave_one_out(
        data,
        key = 'userid',
        target = None,
        sample_top = False,
        random_state = None
    ):
    '''
    Samples 1 item per every user according to the rule `sample_top`.
    It always shuffles the input data. The reason is that even if sampling
    top-rated elements, there could be several items with the same top rating.
    '''
    if sample_top: # sample item with the highest target value (e.g., rating, time, etc.)
        idx = (
            data[target]
            .sample(frac=1, random_state=random_state) # handle same feedback for different items
            .groupby(data[key], sort=False)
            .idxmax()
        ).values
    else: # sample random item
        idx = (
            data[key]
            .sample(frac=1, random_state=random_state)
            .drop_duplicates(keep='first') # data is shuffled - simply take the 1st element
            .index
        ).values

    observed = data.drop(idx)
    holdout = data.loc[idx]
    return observed, holdout

def transform_indices(data, users, items):
    '''
    Reindex columns that correspond to users and items.
    New index is contiguous starting from 0.
    '''
    data_index = {}
    for entity, field in zip(['users', 'items'], [users, items]):
        new_index, data_index[entity] = to_numeric_id(data, field)
        data = data.assign(**{f'{field}': new_index}) # makes a copy of dataset!
    return data, data_index


def to_numeric_id(data, field):
    '''
    Get new contiguous index by converting the data field
    into categorical values.
    '''
    idx_data = data[field].astype("category")
    idx = idx_data.cat.codes
    idx_map = idx_data.cat.categories.rename(field)
    return idx, idx_map


def reindex_data(data, data_index, fields=None):
    '''
    Reindex provided data with the specified index mapping.
    By default, will take the name of the fields to reindex from `data_index`.
    It is also possible to specify which field to reindex by providing `fields`.
    '''
    if fields is None:
        fields = data_index.keys()
    if isinstance(fields, str): # handle single field provided as a string
        fields = [fields]
    for field in fields:
        entity_name = data_index[field].name
        new_index = data_index[field].get_indexer(data[entity_name])
        data = data.assign(**{f'{entity_name}': new_index}) # makes a copy of dataset!
    return data

def reindex(raw_data, index, filter_invalid=True, names=None):
    '''
    Factorizes column values based on provided pandas index. Allows resetting
    index names. Optionally drops rows with entries not present in the index.
    '''
    if isinstance(index, pd.Index):
        index = [index]

    if isinstance(names, str):
        names = [names]

    if isinstance(names, (list, tuple, pd.Index)):
        for i, name in enumerate(names):
            index[i].name = name

    new_data = raw_data.assign(**{
        idx.name: idx.get_indexer(raw_data[idx.name]) for idx in index
    })

    if filter_invalid:
        # pandas returns -1 if label is not present in the index
        # checking if -1 is present anywhere in data
        maybe_invalid = new_data.eval(
            ' or '.join([f'{idx.name} == -1' for idx in index])
        )
        if maybe_invalid.any():
            print(f'Filtered {maybe_invalid.sum()} invalid observations.')
            new_data = new_data.loc[~maybe_invalid]

    return new_data



def verify_time_split(training, holdout):
    '''
    check that holdout items have later timestamps than
    corresponding user's any item from training.
    '''
    holdout_ts = holdout.set_index('userid')['timestamp']
    training_ts = training.groupby('userid')['timestamp'].max()
    assert holdout_ts.ge(training_ts).all()
    
def matrix_from_observations(data, data_description):
    useridx = data[data_description['users']]
    itemidx = data[data_description['items']]
    if 'feedback' in data_description:
        values = data[data_description['feedback']]
    else:
        values = np.ones(len(data))
    return csr_matrix((values, (useridx, itemidx)), dtype='f4')
