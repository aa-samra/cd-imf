import numpy as np
from tqdm import tqdm

class Dataset:
    def __init__(self, training, holdout, description) -> None:
        self.training = training
        self.holdout = holdout
        self.description = description
        self.samples = {}
        self.test_users = self.holdout[self.description['users']].drop_duplicates().values

    def generate_samples(self, 
                         negatives_per_item=999, 
                         item_catalog='absolute',  
                         sampling_method='per_item',
                         exclude_seen_items=False):
        self.samples = []
        item_catalog_size = self.description['n_items']
        if item_catalog=='absolute':
            item_catalog = self.description['n_items']
        elif item_catalog=='relative':
            item_catalog = self.training[self.description['items']].unique()

        for userid, positive_items in tqdm(self.holdout.groupby('userid')['itemid'].agg(list).items()):
            if sampling_method=='per_user':
                sample_size = min((negatives_per_item+1) * len(positive_items), item_catalog_size)
                sampled_items = np.random.choice(item_catalog, size=sample_size ,replace=False)
                # All positive items must be sampled
                for item in positive_items:
                    if (sampled_items==item).any():
                        pass
                    else:
                        # insert it instead of a random negative item
                        while True:
                            rnd_idx = np.random.randint(sample_size)
                            if not sampled_items[rnd_idx] in positive_items:
                                sampled_items[rnd_idx] = item
                                break
                self.samples.append((userid, sampled_items))

            elif sampling_method=='per_item':
                sample_size = min((negatives_per_item+1), item_catalog_size)
                for target_item in positive_items:
                    if exclude_seen_items:
                        seen_items = self.training.query(f'userid == @userid')['itemid'].values
                        forbidden_items = np.concatenate([positive_items, seen_items])
                    else: 
                        forbidden_items = positive_items
                    # sample some redundant items
                    sampled_items = np.random.choice(item_catalog, size=sample_size+len(forbidden_items) ,replace=False)
                    # only one positive item should be sampled
                    for pos_item in forbidden_items:
                        if pos_item!=target_item:  # remove this postive item
                            mask = sampled_items==pos_item
                            if mask.any():
                                sampled_items[mask] = -1


                    # remove (-1)'s elements, and keep the first `sample_size` element 
                    sampled_items = sampled_items[sampled_items>=0]

                    sampled_items = sampled_items[:sample_size]
                    if not (sampled_items==target_item).any():
                        sampled_items[0] = target_item # place it first
                    assert len(sampled_items)==sample_size, f"Length {len(sampled_items)} != {sample_size}"
                    assert (sampled_items==target_item).sum()==1, f"{(sampled_items==target_item).sum()}!=1"
                    assert len(np.intersect1d(sampled_items, forbidden_items))==1, f"{len(np.intersect1d(sampled_items, positive_items))} != 1"
                    self.samples.append((userid, sampled_items))
            else:
                raise(Exception(r'Unknown sampling method, it can be either `per_user` or `per_item`'))