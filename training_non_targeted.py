import argparse
from dataset_loader import DataLoader
from utils import random_planetoid_splits, rand_train_test_idx, get_train_val_test, index_to_mask, normalize_adj_tensor, random_sample_edges, one_hot
from models import *
import torch
import torch.nn.functional as F
from tqdm import tqdm
import random
import seaborn as sns
import numpy as np
from torch_geometric.datasets import Planetoid
import time
from torch_geometric.utils import to_dense_adj, add_self_loops, remove_self_loops, dropout_adj
from torch_geometric.loader import ClusterData



def RunExp(args, dataset, data, Net, rb, val_lb):

    def train(args, model, optimizer, data, dprate, rb):
        model.train()
        optimizer.zero_grad()
        if args.net in ['GCN','GCNII','GAT','SGC_Net','SSGC_Net', 'APPNP', 'GPRGNN']:
            out_main = model(data)[[data.train_mask]]
            nll_main = F.nll_loss(out_main, data.y[data.train_mask])
            loss = nll_main
        elif args.net in ['GCN_mamba_Net']:
            out_main = model(data)
            nll_main = F.nll_loss(out_main[data.train_mask], data.y[data.train_mask])
            loss = nll_main
        elif args.net in ['GCN_mamba_Net_pro_max']:
            out_main = model(args, data)[data.train_mask]
            logits_contrast = torch.matmul(out_main[:,1,:], data.cluster_train_label.t()) / args.tau
            loss_contrast = F.cross_entropy(logits_contrast, data.cluster_train_id)
            nll_main = F.nll_loss(out_main[:,-1,:], data.y[data.train_mask])
            loss = nll_main  + loss_contrast * 0.1
        
        loss.backward()
        optimizer.step()
        del out_main

    def test(args,  model, data, rb):
        model.eval()
        accs, losses, preds = [], [], []
        for _, mask in data('train_mask', 'val_mask', 'test_mask'):
            if args.net in ['GCN', 'GCNII','GAT','SGC_Net','SSGC_Net', 'APPNP', 'GPRGNN']:
                logits_main = model(data)
                pred = logits_main[mask].max(-1)[1]
                acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
            elif args.net in ['GCN_mamba_Net_pro_max']:
                logits_main = model(args, data)[:,-1,:]
                pred = logits_main[mask].max(-1)[1]
                acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
            elif args.net in ['GCN_mamba_Net']:
                logits_main = model(data)
                pred = logits_main[mask].max(-1)[1]
                acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()

            preds.append(pred.detach().cpu())
            accs.append(acc)
        return accs, preds

    tmp_net = Net(dataset, args)

    #randomly split dataset
    if args.dataset in ['Cora_ML', 'Citeseer', 'Pubmed', 'Computers', 'Photo', 'Actor', 'Wisconsin']:

        idx_train = []
        idx_val = []
        idx_test = []
        with open('./data/{}/idx_train_semi-supervised.txt'.format(args.dataset.lower()), 'r') as f:
            for idx in f.readlines():
                idx = idx.strip('\n').split(' ')
                idx_train.append(int(idx[0]))
        with open('./data/{}/idx_val_semi-supervised.txt'.format(args.dataset.lower()), 'r') as f:
            for idx in f.readlines():
                idx = idx.strip('\n').split(' ')
                idx_val.append(int(idx[0]))
        with open('./data/{}/idx_test_semi-supervised.txt'.format(args.dataset.lower()), 'r') as f:
            for idx in f.readlines():
                idx = idx.strip('\n').split(' ')
                idx_test.append(int(idx[0]))
        data.train_mask = index_to_mask(idx_train, data.num_nodes)
        data.val_mask = index_to_mask(idx_val, data.num_nodes)
        data.test_mask = index_to_mask(idx_test, data.num_nodes)

        idx_train = data.train_mask.to(args.device)
        idx_val = data.val_mask.to(args.device)
        idx_test =  data.test_mask.to(args.device)

        data.idx_train = idx_train
        data.idx_val = idx_val
        data.idx_test = idx_test

        labels = data.y.to(args.device)

        cluster_train_id = one_hot(index[idx_train].cpu(), args.cluster).to(args.device)
        cluster_train_label = one_hot(labels[idx_train].cpu(), dataset.num_classes).to(args.device)
        cluster_train_label = torch.mm(cluster_train_id.t(), cluster_train_label)
        cluster_train_label = cluster_train_label / cluster_train_label.sum(1).unsqueeze(1)
        data.cluster_train_label = cluster_train_label
        data.cluster_train_id = cluster_train_id
        
        cluster_val_id = one_hot(index[idx_train+idx_val].cpu(), args.cluster).to(args.device)
        cluster_val_label = one_hot(labels[idx_train+idx_val].cpu(), dataset.num_classes).to(args.device)
        cluster_val_label = torch.mm(cluster_val_id.t(), cluster_val_label)
        cluster_val_label = cluster_val_label / cluster_val_label.sum(1).unsqueeze(1)
        data.cluster_val_label = cluster_val_label
        data.cluster_val_id = cluster_val_id


    model, data = tmp_net.to(args.device), data.to(args.device)

    
    optimizer = torch.optim.Adam(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)

    best_val_acc = test_acc = 0
    best_val_loss = float('inf')
    val_loss_history = []
    val_acc_history = []
    time_run=[]

    for epoch in range(args.epochs):
        t_st=time.time()
        train(args, model, optimizer, data, args.dprate, rb)
        time_epoch=time.time()-t_st  # each epoch train times
        time_run.append(time_epoch)
        [train_acc, val_acc, tmp_test_acc], preds = test(args, model, data, rb) # [train_loss, val_loss, tmp_test_loss]
        
        # if val_loss < best_val_loss:
        if best_val_acc < val_acc:
            best_val_acc = val_acc
            # best_val_loss = val_loss
            test_acc = tmp_test_acc
            if args.net =='BernNet':
                TEST = tmp_net.prop1.temp.clone()
                theta = TEST.detach().cpu()
                theta = torch.relu(theta).numpy()
            else:
                theta = args.alpha


        if epoch >= 0:
            # val_loss_history.append(val_loss)
            val_acc_history.append(val_acc)
            if args.early_stopping > 0 and epoch > args.early_stopping:
                tmp = torch.tensor(
                    val_acc_history[-(args.early_stopping + 1):-1])
                if val_acc < tmp.mean().item():
                    print('The sum of epochs:',epoch)
                    break
        print('train_acc:{},val_acc:{},temp_test_acc:{}'.format(train_acc, val_acc, tmp_test_acc))
    return test_acc, best_val_acc, theta, time_run, model

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

#adj normalization
def adj_nor(edge):
    degree = torch.sum(edge, dim=1)
    degree = 1 / torch.sqrt(degree)
    degree = torch.diag(degree)
    adj = torch.mm(torch.mm(degree, edge), degree)
    return adj


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=15, help='seeds for random splits.')
    parser.add_argument('--epochs', type=int, default=5000, help='max epochs.')
    parser.add_argument('--lr', type=float, default=0.05, help='learning rate.')     
    parser.add_argument('--Stat_lr', type=float, default=0.1, help='State learning rate.')   
    parser.add_argument('--weight_decay', type=float, default=5e-5, help='weight decay.')  
    parser.add_argument('--early_stopping', type=int, default=200, help='early stopping.')
    parser.add_argument('--hidden', type=int, default=64, help='hidden units.')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout for neural networks.')

    parser.add_argument('--lamda', type=float, default=0.5, help='propagation steps.')
    parser.add_argument('--weight_decay1', type=float, default=5e-4, help='weight decay.')
    parser.add_argument('--weight_decay2', type=float, default=5e-4, help='weight decay.')


    parser.add_argument('--train_rate', type=float, default=0.6, help='train set rate.')
    parser.add_argument('--val_rate', type=float, default=0.2, help='val set rate.')
    parser.add_argument('--K', type=int, default=10, help='propagation steps.')
    parser.add_argument('--alpha', type=float, default=0.5, help='alpha for APPN/GPRGNN.')
    parser.add_argument('--dprate', type=float, default=0.0, help='dropout for propagation layer.')
    parser.add_argument('--Init', type=str,choices=['SGC', 'PPR', 'NPPR', 'Random', 'WS', 'Null'], default='PPR', help='initialization for GPRGNN.')
    parser.add_argument('--heads', default=8, type=int, help='attention heads for GAT.')
    parser.add_argument('--output_heads', default=1, type=int, help='output_heads for GAT.')

    parser.add_argument('--dataset', type=str, choices=['Cora_ML', 'Polblogs','Wisconsin', 'Cora','Citeseer','Pubmed','Computers','Photo','Chameleon','Squirrel','Actor','Texas','Cornell',
                                                        'Roman-empire', 'Amazon-ratings', 'Minesweeper', 'Tolokers', 'Questions'],
                        default='Cora')
    parser.add_argument('--device', type=int, default=1, help='GPU device.')
    parser.add_argument('--runs', type=int, default=1, help='number of runs.')
    parser.add_argument('--net', type=str, choices=['GCN_mamba_Net_pro_max','SSGC_Net', 'SGC_Net', 'GCNII', 'GCN_mamba_Net', 'GCN_mamba_Net_plus', 'GCN', 'GAT', 'APPNP', 'ChebNet', 'GPRGNN','BernNet','MLP'], default='BernNet')
    parser.add_argument('--Bern_lr', type=float, default=0.01, help='learning rate for BernNet propagation layer.')

    # parameters for mamba
    parser.add_argument('--tau', type=float, default=1.0, help='softmax tempurate.')
    parser.add_argument('--d_model', type=int, default=64, help='hidden units.')
    parser.add_argument('--cluster', type=int, default=5, help='hidden units.')
    parser.add_argument('--d_inner', type=int, default=64, help='')
    parser.add_argument('--dt_rank', type=int, default=4, help='')
    parser.add_argument('--d_state', type=int, default=4, help='')
    parser.add_argument('--bias', type=bool, default=False, help='')
    parser.add_argument('--mamba_dropout', type=float, default=0.6, help='')
    parser.add_argument('--layer_num', type=int, default=3, help='')
    parser.add_argument('--d_conv', type=int, default=4, help='')

    args = parser.parse_args()
    print(args)
    print("---------------------------------------------")
    dataset = DataLoader(args.dataset)
    data = dataset[0]

    args.device = torch.device('cuda:'+str(args.device) if torch.cuda.is_available() else 'cpu')
    modified_adj = np.loadtxt('./attacked_data_for_MbaGCN/GraD_modified_{}_0.05.txt'.format(args.dataset.lower()),delimiter=' ')
    modified_adj = torch.FloatTensor(modified_adj).to(args.device)
    
    row, col = (modified_adj > 0).nonzero(as_tuple=True)
    edge_index = torch.stack([row, col], dim=0)
    data.edge_index = edge_index
    val = modified_adj[row, col]
    N = modified_adj.size(0)
    adj_t = SparseTensor(row=row, col=col, value=val, sparse_sizes=(N, N))
    adj_t = adj_t.set_diag()
    deg = adj_t.sum(dim=1).to(torch.float)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    adj_t = deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)
    data.adj_t = adj_t.to_dense()

    cluster_data = ClusterData(data, num_parts=args.cluster, recursive=False)
    index = torch.zeros(data.x.shape[0])
    for i in range(args.cluster):
        index[cluster_data.partition.node_perm[cluster_data.partition.partptr[i]:cluster_data.partition.partptr[i+1]]] = i
    index = index.long().to(args.device)
    data.index = index


    percls_trn = int(round(args.train_rate*len(data.y)/dataset.num_classes))
    val_lb = int(round(args.val_rate*len(data.y)))

    results = []
    thetas = []
    time_results=[]
    for RP in tqdm(range(args.runs)):
        gnn_name = args.net
        if gnn_name =='GCN_mamba_Net':
            Net = GCN_mamba_Net
        elif gnn_name =='GCN_mamba_Net_pro_max':
            Net = GCN_mamba_Net_pro_max

        test_acc, best_val_acc, theta_0, time_run,tmp_net = RunExp(args, dataset, data, Net, RP, val_lb)
        time_results.append(time_run)
        results.append([test_acc, best_val_acc])
        thetas.append(theta_0)
        print(f'run_{str(RP+1)} \t test_acc: {test_acc:.4f}')

    run_sum=0
    epochsss=0
    for i in time_results:
        run_sum+=sum(i)
        epochsss+=len(i)

    print("each run avg_time:",run_sum/(args.runs),"s")
    print("each epoch avg_time:",1000*run_sum/epochsss,"ms")
    test_acc_mean, val_acc_mean = np.mean(results, axis=0) * 100
    test_acc_std = np.sqrt(np.var(results, axis=0)[0]) * 100

    values=np.asarray(results)[:,0]
    uncertainty=np.max(np.abs(sns.utils.ci(sns.algorithms.bootstrap(values,func=np.mean,n_boot=1000),95)-values.mean()))

    print(f'{gnn_name} on dataset {args.dataset}, in {args.runs} repeated experiment:')
    print(f'test acc mean = {test_acc_mean:.2f}±{uncertainty*100:.2f}  \t val acc mean = {val_acc_mean:.2f}')
