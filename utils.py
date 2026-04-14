import torch
import math
import numpy as np
from sklearn.model_selection import train_test_split


def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool)
    mask[index] = 1
    return mask

def random_planetoid_splits(data, num_classes, percls_trn=20, val_lb=500, seed=12134):
    index=[i for i in range(0,data.y.shape[0])]
    train_idx=[]
    rnd_state = np.random.RandomState(seed)
    for c in range(num_classes):
        class_idx = np.where(data.y.cpu() == c)[0]
        if len(class_idx)<percls_trn:
            train_idx.extend(class_idx)
        else:
            train_idx.extend(rnd_state.choice(class_idx, percls_trn,replace=False))
    rest_index = [i for i in index if i not in train_idx]
    val_idx=rnd_state.choice(rest_index,val_lb,replace=False)
    test_idx=[i for i in rest_index if i not in val_idx]
    #print(test_idx)

    data.train_mask = index_to_mask(train_idx,size=data.num_nodes)
    data.val_mask = index_to_mask(val_idx,size=data.num_nodes)
    data.test_mask = index_to_mask(test_idx,size=data.num_nodes)
    
    return data

def rand_train_test_idx(data, train_prop=.6, valid_prop=.2, seed=15):
    """ randomly splits label into train/valid/test splits """
    # import ipdb;ipdb.set_trace()
    index=[i for i in range(0, data.y.shape[0])]
    n = data.y.shape[0]
    rnd_state = np.random.RandomState(seed)
    train_idx = rnd_state.choice(index, int(n * train_prop), replace=False)
    rest_index = [i for i in index if i not in train_idx]
    val_idx=rnd_state.choice(rest_index,int(n * valid_prop),replace=False)
    test_idx=[i for i in rest_index if i not in val_idx]

    data.train_mask = index_to_mask(train_idx,size=data.num_nodes)
    data.val_mask = index_to_mask(val_idx,size=data.num_nodes)
    data.test_mask = index_to_mask(test_idx,size=data.num_nodes)

    return data

def get_train_val_test(data, idx, train_size, val_size, test_size, stratify):

    idx_train_and_val, idx_test = train_test_split(idx,
                                                   random_state=None,
                                                   train_size=train_size + val_size,
                                                   test_size=test_size,
                                                   stratify=stratify)

    if stratify is not None:
        stratify = stratify[idx_train_and_val]

    idx_train, idx_val = train_test_split(idx_train_and_val,
                                          random_state=None,
                                          train_size=(train_size / (train_size + val_size)),
                                          test_size=(val_size / (train_size + val_size)),
                                          stratify=stratify)

    data.train_mask = index_to_mask(idx_train,size=data.num_nodes)
    data.val_mask = index_to_mask(idx_val,size=data.num_nodes)
    data.test_mask = index_to_mask(idx_test,size=data.num_nodes)
    return data

def normalize_adj_tensor(adj):
    mx = adj
    rowsum = mx.sum(1)
    r_inv = rowsum.pow(-1/2).flatten()
    r_inv[torch.isinf(r_inv)] = 0.
    r_mat_inv = torch.diag(r_inv)
    mx = r_mat_inv @ mx
    mx = mx @ r_mat_inv
    return mx

def random_sample_edges(adj, n, exclude):
        itr = sample_forever(adj, exclude=exclude)
        return [next(itr) for _ in range(n)]

def sample_forever(adj, exclude):
    """Randomly random sample edges from adjacency matrix, `exclude` is a set
    which contains the edges we do not want to sample and the ones already sampled
    """
    while True:
        # t = tuple(np.random.randint(0, adj.shape[0], 2))
        # t = tuple(random.sample(range(0, adj.shape[0]), 2))
        t = tuple(np.random.choice(adj.shape[0], 2, replace=False))
        if t not in exclude:
            yield t
            exclude.add(t)
            exclude.add((t[1], t[0]))

def one_hot(x, class_count):
    res = torch.nn.functional.one_hot(x, class_count)
    res = res.float()
    return res
