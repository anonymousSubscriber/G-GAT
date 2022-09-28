

import time
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from process import *
from utils import *
from model import *
# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42, help='Random seed.') #Default seed same as GCNII
parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs to train.')
parser.add_argument('--layers', type=int, default=3, help='Number of layers.')
parser.add_argument('--nhid', type=int, default=64, help='hidden dimensions.')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
parser.add_argument('--patience', type=int, default=100, help='Patience')
parser.add_argument('--data', default='chameleon', help='dateset')
parser.add_argument('--dev', type=int, default=0, help='device id')
parser.add_argument('--w_att',type=float, default=0.0005, help='Weight decay scalar')
parser.add_argument('--w_fc2',type=float, default=0.0005, help='Weight decay layer-2')
parser.add_argument('--w_fc1',type=float, default=0.0005, help='Weight decay layer-1')
parser.add_argument('--lr_fc',type=float, default=0.02, help='Learning rate 2 fully connected layers')
parser.add_argument('--lr_att',type=float, default=0.02, help='Learning rate Scalar')
parser.add_argument('--self_loop',type=int, default=0, help='Type of features to be used')
# parser.add_argument('--layern',type=int, default=0, help='Type of features to be used')

args = parser.parse_args()
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
# print("==========================")
# print(f"Dataset: {args.data}, method: {args.method} model: {args.model_method} x,y= {args.ratio_x},{args.ratio_y} r_type: {args.r_type}")
# print(f"Dropout:{args.dropout}, layer_norm: {layer_norm}")
# print(f"w_att:{args.w_att}, w_fc2:{args.w_fc2}, w_fc1:{args.w_fc1}, lr_fc:{args.lr_fc}, lr_att:{args.lr_att}")


cudaid = "cuda:"+str(args.dev)
device = torch.device(cudaid)
checkpt_file = 'pretrained/best.pt'



    
def train_step(model,optimizer,labels,model_input,idx_train):
    model.train()
    optimizer.zero_grad()
    output = model(model_input)
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()
    return loss_train.item(),acc_train.item()


def validate_step(model,labels,model_input,idx_val):
    model.eval()
    with torch.no_grad():
        output = model(model_input)
        loss_val = F.nll_loss(output[idx_val], labels[idx_val])
        acc_val = accuracy(output[idx_val], labels[idx_val])
        return loss_val.item(),acc_val.item()

def test_step(model,labels,model_input,idx_test):
    model.load_state_dict(torch.load(checkpt_file))
    model.eval()
    with torch.no_grad():
        output = model(model_input)
        loss_test = F.nll_loss(output[idx_test], labels[idx_test])
        acc_test = accuracy(output[idx_test], labels[idx_test])
        return loss_test.item(),acc_test.item()


def train(datastr,splitstr):
    adj, adj_i, features, labels, idx_train, idx_val, idx_test, num_features, num_labels = full_load_data(datastr,splitstr)
    adj=adj.to_dense()
    adj_i=adj_i.to_dense()
    model_input=features.to(device)

    if args.self_loop==1:
        adj_input=adj_i
    else:
        adj_input=adj
    
    # adj_input=mask_graph(adj_input,args.method,(args.ratio_x,args.ratio_y),labels,args.r_type)
    labels=labels.to(device)

    adj_input=adj_input.to(device)

    
    # model ,optimizer=build_model(args,num_features,num_labels,device,adj_input)
    adj_list = []
    adj_list.append(None)
    no_loop_mat = torch.eye(adj_input.shape[0]).to(device)

    for ii in range(args.layers):
        no_loop_mat = torch.mm(adj_input, no_loop_mat)
        adj_list.append(no_loop_mat)

    model = GGAT(nfeat=num_features,
            nhid=args.nhid,
            nclass=num_labels,
            dropout=args.dropout,
            adj_list=adj_list,
            adj_att=adj_input).to(device)
    optimizer_sett = [
    {'params': model.classifier.parameters(), 'weight_decay': args.w_fc2, 'lr': args.lr_fc},
    {'params': model.fc1.parameters(), 'weight_decay': args.w_fc1, 'lr': args.lr_fc},
    {'params': model.hop_select, 'weight_decay': args.w_att, 'lr': args.lr_att},
    {'params': model.Q.parameters(), 'weight_decay': args.w_fc1/4, 'lr': args.lr_fc},
    {'params': model.K.parameters(), 'weight_decay': args.w_fc1/4, 'lr': args.lr_fc},
    {'params': model.w, 'weight_decay': 0.0, 'lr': args.lr_fc},
    ]
    optimizer = optim.Adam(optimizer_sett)

    bad_counter = 0
    best = 999999999
    for epoch in range(args.epochs):
        loss_tra,acc_tra = train_step(model,optimizer,labels,model_input,idx_train)
        loss_val,acc_val = validate_step(model,labels,model_input,idx_val)
        #Uncomment following lines to see loss and accuracy values
        '''
        if(epoch+1)%1 == 0:

            print('Epoch:{:04d}'.format(epoch+1),
                'train',
                'loss:{:.3f}'.format(loss_tra),
                'acc:{:.2f}'.format(acc_tra*100),
                '| val',
                'loss:{:.3f}'.format(loss_val),
                'acc:{:.2f}'.format(acc_val*100))
        '''        

        if loss_val < best:
            best = loss_val
            torch.save(model.state_dict(), checkpt_file)
            bad_counter = 0
        else:
            bad_counter += 1

        if bad_counter == args.patience:
            break
    test_out = test_step(model,labels,model_input,idx_test)
    acc = test_out[1]


    return acc*100

t_total = time.time()
acc_list = []

for i in range(10):
    datastr = args.data
    splitstr = 'splits/'+args.data+'_split_0.6_0.2_'+str(i)+'.npz'
    accuracy_data = train(datastr,splitstr)
    acc_list.append(accuracy_data)


    ##print(i,": {:.2f}".format(acc_list[-1]))

# print("Train cost: {:.4f}s".format(time.time() - t_total))
#print("Test acc.:{:.2f}".format(np.mean(acc_list)))
# print(f"Test accuracy: {np.round(np.mean(acc_list),2)}, {np.round(np.std(acc_list),2)}")
print(f"{np.round(np.mean(acc_list),2)}, {np.round(np.std(acc_list),2)} |",end=' ')
