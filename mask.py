from certifi import where
import torch
import random
def to_one(adj):
    return torch.where(adj>0,torch.ones_like(adj),torch.zeros_like(adj))
def norm(adj):
    adj=to_one(adj).float()
    degree=torch.sum(adj,dim=-1)
    degree[degree==0]=1
    degree=degree**(-0.5)
    D=torch.diag(degree)
    return torch.mm(torch.mm(D,adj),D)
def adj2vector(adj):
    node_number=adj.shape[0]
    res=[]
    for i in range(node_number):
        res.append(adj[i].nonzero().flatten(0,-1).tolist())
    return res
def get2hop(adj):
    '''
    get 2 hop edges but not include 1hop edges
    '''
    proto=torch.mm(adj,adj)-1000*adj
    return torch.where(proto>0,torch.ones_like(adj),torch.zeros_like(adj))
def vector2adj(edges):
    node_number=len(edges)
    res=torch.zeros((node_number,node_number))
    for u in range(node_number):
        for v in edges[u]:
            res[u][v]=1
    return res
def replace(adj,adj_candidate,x=0,y=0):
    node_number=adj.shape[0]
    rand_tensor=torch.rand((node_number,))
    edges1=adj2vector(adj)
    edges2=adj2vector(adj_candidate)
    for i in range(node_number):
        if rand_tensor[i]<=x:
            edges_number=len(edges1[i])
            replaced=torch.rand((edges_number,))
            random.shuffle(edges2[i])
            for j in range(edges_number):
                if replaced[j]<=y and len(edges2[i])>0:
                    edges1[i][j]=edges2[i][0]
                    edges2[i].pop(0)

    return vector2adj(edges1)
def mask_graph(adj,method,ratio,labels,r_type=0):
    '''
    adj: adjcent matrix
    method:
        0: don't mask
        1: mask edge
        2: replace edges with 2 hop edges
        3: repalce edges with random edge
    ratio:
        pair<float,float> (x,y) x of node's edges will be masked with y probability
    '''
    if method==0:# protograph
        adj=norm(adj)
        return adj
    elif method==1: 
        '''
        drop edge 
        '''
        node_number=adj.shape[0]
        rand_tensor=torch.rand((node_number,))
        node_masked=torch.where(rand_tensor<ratio[0],torch.ones_like(rand_tensor),torch.zeros_like(rand_tensor))
        node_masked=node_masked.reshape(-1,1).repeat(1,node_number)
        labels=labels.reshape(-1,1)
        t_mask=labels-labels.T
        if r_type==0:
            t_mask=torch.where(t_mask==0,torch.ones_like(adj),torch.zeros_like(adj))
        else:
            t_mask=torch.where(t_mask==0,torch.zeros_like(adj),torch.ones_like(adj))

        adj=to_one(adj)
        mask=torch.where(torch.rand_like(adj)<=ratio[1],torch.ones_like(adj),torch.zeros_like(adj))        
        mask_edge=adj*t_mask*mask*node_masked
        adj=adj-mask_edge
        adj=norm(adj)
        return adj
    elif method==2:
        '''
        replace
        ratio (x,y) x% nodes' y% homogeneous edges was replace by hetergeneous edges
            from 2 hop 
        '''
        # print("replace the edges with 2hop edges")
        labels=labels.reshape(-1,1)
        heter_mask=labels-labels.T
        heter_mask=torch.where(heter_mask==0,torch.zeros_like(adj),torch.ones_like(adj))
        homo_mask=torch.where(heter_mask==0,torch.ones_like(adj),torch.zeros_like(adj))
        homo_adj=adj*homo_mask
        heter_adj=adj*heter_mask 
        adj2=get2hop(adj)
        homo_adj2=adj2*homo_mask
        heter_adj2=adj2*heter_mask
        if r_type==0:#replace homo
            new_sub_adj=replace(homo_adj,heter_adj2,ratio[0],ratio[1])
            return norm(heter_adj+new_sub_adj)
        else:
            new_sub_adj=replace(heter_adj,homo_adj2,ratio[0],ratio[1])
            return norm(homo_adj+new_sub_adj)
    elif method==3:
        # print("replace the edges with all candidate edges")
        labels=labels.reshape(-1,1)
        heter_mask=labels-labels.T
        heter_mask=torch.where(heter_mask==0,torch.zeros_like(adj),torch.ones_like(adj))
        homo_mask=torch.where(heter_mask==0,torch.ones_like(adj),torch.zeros_like(adj))
        homo_adj=adj*homo_mask
        heter_adj=adj*heter_mask 
        if r_type==0:#replace homo
            new_sub_adj=replace(homo_adj,to_one(heter_mask-adj*10),ratio[0],ratio[1])
            return norm(heter_adj+new_sub_adj)
        else:
            new_sub_adj=replace(heter_adj,to_one(homo_mask-adj*10),ratio[0],ratio[1])
            return norm(homo_adj+new_sub_adj)
    else:
        print("don't have that method to mask node")
        raise