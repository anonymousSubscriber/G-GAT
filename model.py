from turtle import forward
import torch.nn as nn
import torch
import math
import numpy as np
import torch.nn.functional as F
import torch.optim as optim

class GCN1(nn.Module):
    def __init__(self,input_dim,output_dim,adj,act='None'):
        super(GCN1,self).__init__()
        self.fc=nn.Linear(input_dim,output_dim)
        self.adj=adj
        self.act=act
    def forward(self,x):
        x=self.fc(x)
        x=torch.mm(self.adj,x)
        if self.act !='None':
            x=F.relu(x)
        return x
def build_QK(input_dim,output_dim,build_att,att_act,adj):
    if build_att=='GCN1':
       return GCN1(input_dim,output_dim,adj,att_act)
    elif build_att=='GCN2':
        pass
    elif build_att=='GAT1':
        pass
    elif build_att=='GAT2':
        pass
    elif build_att=='GBP':
        pass
    elif build_att=="SGC":
        pass
    elif build_att=='linear':
        pass
    else:
        raise

class GGAT(nn.Module):
    def __init__(self,nfeat,nhid,nclass,adj_list,dropout,adj_att,
            Q_method='GCN1',Q_act='None',K_method='GCN1',K_act='None'):
        super(GGAT,self).__init__()
        self.nlayers=len(adj_list)
        self.classifier = nn.Linear((nhid)*self.nlayers,nclass)
        self.adj_list=adj_list
        self.dropout = dropout
        self.act_fn = nn.ReLU()
        self.fc1 = nn.ModuleList([nn.Linear(nfeat,nhid) for _ in range(self.nlayers)])
        self.Q = nn.ModuleList([build_QK(nhid,nhid,Q_method,Q_act,adj_att) for _ in range(self.nlayers-1)])
        self.K = nn.ModuleList([build_QK(nhid,nhid,K_method,K_act,adj_att) for _ in range(self.nlayers-1)])

        self.hop_select = nn.Parameter(torch.ones(self.nlayers))
        self.w=nn.Parameter(torch.ones(size=(self.nlayers-1,2)))


    def _soft_max_att(self,adj,attention):
        attention=torch.where(adj>0,attention,torch.ones_like(attention)*-9e15)
        return F.softmax(attention,dim=-1)

    def forward(self,input):
        x=input
        mask = F.softmax(self.hop_select,dim=-1)
        list_out = list()
        for i in range(self.nlayers):
            tmp_out =self.fc1[i](x)
            if self.adj_list[i] is not None:
                
                W_att=self.sm(self.w[i-1])

                tmp_out_att=torch.mm(self.adj_att,tmp_out)
                Q=self.Q[i-1](tmp_out_att)
                K=self.K[i-1](tmp_out_att)
                attention=self._soft_max_att(self.adj_list[i],torch.mm(Q,K.T))*W_att[0] \
                    +self.adj_list[i]*W_att[1]
                tmp_out=torch.mm(attention,tmp_out)

            tmp_out = F.normalize(tmp_out,p=2,dim=1)
            tmp_out = torch.mul(mask[i],tmp_out)
            list_out.append(tmp_out)

        final_mat = torch.cat(list_out, dim=1)
        out = self.act_fn(final_mat)
        out = F.dropout(out,self.dropout,training=self.training)
        out = self.classifier(out)


        return F.log_softmax(out, dim=1)



# def build_model(args,num_features,num_labels,device,adj):
#     if args.model_method=='gcn':
#         model = GCN(nfeat=num_features,
#                 nhid=args.nhid,
#                 nclass=num_labels,adj=adj).to(device)


#         optimizer_sett = [
#         {'params': model.classifier.parameters(), 'weight_decay': args.w_fc2, 'lr': args.lr_fc},
#         {'params': model.fc.parameters(), 'weight_decay': args.w_fc1, 'lr': args.lr_fc},
#         ]
#         optimizer = optim.Adam(optimizer_sett)
#         return model,optimizer
#     if args.model_method=='gat3':
#         adj_list = []
#         adj_list.append(None)
#         no_loop_mat = torch.eye(adj.shape[0]).to(device)

#         for ii in range(args.nlayers):
#             no_loop_mat = torch.mm(adj, no_loop_mat)
#             adj_list.append(no_loop_mat)

#         model = FSGNN(nfeat=num_features,
#                 nhid=args.nhid,
#                 nclass=num_labels,
#                 dropout=args.dropout,
#                 adj_list=adj_list,
#                 adj_att=adj).to(device)


#         optimizer_sett = [
#         {'params': model.fc2.parameters(), 'weight_decay': args.w_fc2, 'lr': args.lr_fc},
#         {'params': model.fc1.parameters(), 'weight_decay': args.w_fc1, 'lr': args.lr_fc},
#         {'params': model.att, 'weight_decay': args.w_att, 'lr': args.lr_att},
#         {'params': model.Q.parameters(), 'weight_decay': args.w_fc1/4, 'lr': args.lr_fc},
#         {'params': model.K.parameters(), 'weight_decay': args.w_fc1/4, 'lr': args.lr_fc},
#         {'params': model.w, 'weight_decay': 0.0, 'lr': args.lr_fc},
#         ]

#         optimizer = optim.Adam(optimizer_sett)
#         return model,optimizer
#     if args.model_method=='gat':
#         model = GAT(nfeat=num_features,
#                 nhid=args.nhid,
#                 nclass=num_labels,adj=adj).to(device)


#         optimizer_sett = [
#         {'params': model.classifier.parameters(), 'weight_decay': args.w_fc2, 'lr': args.lr_fc},
#         {'params': model.fc.parameters(), 'weight_decay': args.w_fc1, 'lr': args.lr_fc},
#         {'params': model.a1.parameters(), 'weight_decay': args.w_fc1, 'lr': args.lr_fc},
#         {'params': model.a2.parameters(), 'weight_decay': args.w_fc1, 'lr': args.lr_fc}
#         ]
#         optimizer = optim.Adam(optimizer_sett)
#         return model,optimizer
#     else:
#         print("can't find that model")
#         raise