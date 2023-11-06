import torch
import torch.nn as nn

import sys

from lpips import pretrained_networks as pn
from lpips import normalize_tensor

import math
import os

def spatial_average(in_tens, keepdim=True):
    return in_tens.mean([2,3],keepdim=keepdim)

class NetLinLayer(nn.Module):
    ''' A single linear layer which does a 1x1 conv '''
    def __init__(self, chn_in, chn_out=1, use_dropout=False):
        super(NetLinLayer, self).__init__()

        layers = [nn.Dropout(),] if(use_dropout) else []
        layers += [nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False),]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
    
class ScalingLayer(nn.Module):
    def __init__(self):
        super(ScalingLayer, self).__init__()
        self.register_buffer('shift', torch.Tensor([-.030,-.088,-.188])[None,:,None,None])
        self.register_buffer('scale', torch.Tensor([.458,.448,.450])[None,:,None,None])

    def forward(self, inp):
        return (inp - self.shift) / self.scale
    
class Dist2LogitLayer(nn.Module):
    ''' takes 2 distances, puts through fc layers, spits out value between [0,1] (if use_sigmoid is True) '''
    def __init__(self, chn_mid=32, use_sigmoid=True):
        super(Dist2LogitLayer, self).__init__()

        layers = [nn.Linear(5, chn_mid, bias=True),]
        layers += [nn.LeakyReLU(0.2,True),]
        layers += [nn.Linear(chn_mid, chn_mid, bias=True),]
        layers += [nn.LeakyReLU(0.2,True),]
        layers += [nn.Linear(chn_mid, 1, bias=True),]
        if(use_sigmoid):
            layers += [nn.Sigmoid(),]
        self.model = nn.Sequential(*layers)

    def forward(self,d0,d1,eps=0.1):
        return self.model.forward(torch.cat((d0,d1,d0-d1,d0/(d1+eps),d1/(d0+eps)),dim=1))

class BCERankingLoss(nn.Module):
    def __init__(self, chn_mid=32):
        super(BCERankingLoss, self).__init__()
        self.net = Dist2LogitLayer(chn_mid=chn_mid)
        # self.parameters = list(self.net.parameters())
        self.loss = torch.nn.BCELoss()

    def forward(self, d0, d1, judge):
        per = (judge+1.)/2.
        self.logit = self.net.forward(d0,d1)
        return self.loss(self.logit, per)

import torch.nn.init as init
class FeatureScalingLayer(nn.Module):
    def __init__(self, dim, use_dropout=False):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(dim, 1, 1))
        self.kernel_size=(1,1)
        
        self.use_dropout = use_dropout
        if self.use_dropout:
            self.dropout = nn.Dropout()
        
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
    
    def forward(self, x):
        if self.use_dropout:
            x = self.dropout(x)
        return x * self.weight
    
class EmbeddingModel(nn.Module):
    def __init__(self, net, use_dropout=True):
        super().__init__()
        if(net in ['vgg','vgg16']):
            net_type = pn.vgg16
            self.chns = [64,128,256,512,512]
        elif(net=='alex'):
            net_type = pn.alexnet
            self.chns = [64,192,384,256,256]
        elif(net=='squeeze'):
            net_type = pn.squeezenet
            self.chns = [64,128,256,384,384,512,512]
        self.L = len(self.chns)
        self.out_features = sum(self.chns)

        self.net = net_type(pretrained=True, requires_grad=False)

        self.lin0 = FeatureScalingLayer(self.chns[0], use_dropout=use_dropout)
        self.lin1 = FeatureScalingLayer(self.chns[1], use_dropout=use_dropout)
        self.lin2 = FeatureScalingLayer(self.chns[2], use_dropout=use_dropout)
        self.lin3 = FeatureScalingLayer(self.chns[3], use_dropout=use_dropout)
        self.lin4 = FeatureScalingLayer(self.chns[4], use_dropout=use_dropout)
        self.lins = [self.lin0,self.lin1,self.lin2,self.lin3,self.lin4]
        if(net=='squeeze'): # 7 layers for squeezenet
            self.lin5 = FeatureScalingLayer(self.chns[5], use_dropout=use_dropout)
            self.lin6 = FeatureScalingLayer(self.chns[6], use_dropout=use_dropout)
            self.lins+=[self.lin5,self.lin6]
        self.lins = nn.ModuleList(self.lins)
        self.scaling_layer = ScalingLayer()
        
    def forward(self, in0, normalize=False):
        if normalize: # turn on this flag if input is [0,1] so it can be adjusted to [-1, +1]
            in0 = 2 * in0  - 1

        # v0.0 - original release had a bug, where input was not scaled
        in0_input = self.scaling_layer(in0) 
        outs0 = self.net.forward(in0_input)
        feats0 = {} 

        for kk in range(self.L):
            feats0[kk] = normalize_tensor(outs0[kk])
            
        res = [spatial_average(self.lins[kk](feats0[kk]), keepdim=False) for kk in range(self.L)]
        
        outs = torch.cat(res, dim=1)

        return outs
    


import wandb
import tqdm
from data import data_loader as dl


#["val/color"],["val/deblur"],["val/frameinterp"],["val/superres"]
class Trainer():
    def __init__(self, batch_size, lr, savedir, outdir, device=torch.device("cuda"), beta1=0.5, nThreads=1, dataroot="./dataset",
                 train_datasets=['train/traditional','train/cnn','train/mix'], 
                 val_datasets=[["val/traditional"],["val/cnn"]]):
        self.model = EmbeddingModel('vgg').to(device)
        self.loss = BCERankingLoss().to(device)
        self.batch_size = batch_size
        self.device = device
        self.savedir = savedir
        self.outdir = outdir
        
        if not os.path.exists(self.savedir):
            os.makedirs(self.savedir)
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)
        
        parameters = list(self.model.lins.parameters()) + list(self.loss.parameters())
        self.opt = torch.optim.Adam(parameters, lr=lr, betas=(beta1, 0.999))
        
        train_loader = dl.CreateDataLoader(train_datasets, dataroot=dataroot, dataset_mode='2afc', 
                                           batch_size=batch_size, serial_batches=False, nThreads=nThreads)
        self.train_dataset = train_loader.load_data()
        
        self.val_datasets={}
        for val_name in val_datasets:
            val_loader = dl.CreateDataLoader(val_name, dataroot=dataroot, dataset_mode='2afc', 
                                             batch_size=batch_size, serial_batches=False, nThreads=nThreads)
            val_dataset = val_loader.load_data()
            self.val_datasets[val_name[0]] = val_dataset
        
    def clamp_weights(self):
        for module in self.model.modules():
            if(hasattr(module, 'weight') and module.kernel_size==(1,1)):
                module.weight.data = torch.clamp(module.weight.data,min=0)
                
    def _forward(self, xref, x0, x1):
        eref = self.model(xref)
        e0 = self.model(x0)
        e1 = self.model(x1)
        
        d0 = (eref - e0).norm(dim=1, keepdim=True)
        d1 = (eref - e1).norm(dim=1, keepdim=True)
        
        return d0, d1
        
    def train_step(self, xref, x0, x1, l):
        d0, d1 = self._forward(xref, x0, x1)
        
        l = l.view(d0.size())
        
        acc = self.compute_accuracy(d0, d1, l)
        
        loss = self.loss(d0, d1, 2*l-1).mean()
        
        return loss, acc
    
    def compute_accuracy(self, d0, d1, l):
        ''' d0, d1 are Variables, judge is a Tensor '''
        d1_lt_d0 = (d1<d0).cpu().data.numpy().flatten()
        judge_per = l.cpu().numpy().flatten()
        return d1_lt_d0*judge_per + (1-d1_lt_d0)*(1-judge_per)
    
    def update_learning_rate(self,nepoch_decay):
        lrd = self.lr / nepoch_decay
        lr = self.old_lr - lrd

        for param_group in self.optimizer_net.param_groups:
            param_group['lr'] = lr

        print('update lr [%s] decay: %f -> %f' % (type,self.old_lr, lr))
        self.old_lr = lr
        
        
    def save(self, path, step):
        savedict = {
            'model': self.model.state_dict(),
            'loss': self.loss.net.state_dict(),
            'step': step
        }
        torch.save(savedict, path)
        
    def load(self, path):
        loaddict = torch.load(path)
        self.model.load_state_dict(loaddict['model'])
        self.loss.net.load_state_dict(loaddict['loss'])
        step = loaddict['step']
        return step
    
    def validate(self, dataset):
        N = len(dataset) * self.batch_size
        d0_all = torch.zeros(N, 1)
        d1_all = torch.zeros(N, 1)
        l_all = torch.zeros(N, 1)
        
        val_loss = 0
        with torch.no_grad():
            for i, data in enumerate(dataset):
                xref, x0, x1, l = data['ref'], data['p0'], data['p1'], data['judge']
                xref, x0, x1, l = xref.to(self.device), x0.to(self.device), x1.to(self.device), l.to(self.device)
                d0, d1 = self._forward(xref, x0, x1)
                l = l.view(d0.size())
                
                val_loss += self.loss(d0, d1, 2*l-1).sum()
                
                jmin, jmax = i * self.batch_size, i * self.batch_size + l.size(0)
                d0_all[jmin:jmax] = d0
                d1_all[jmin:jmax] = d1
                l_all[jmin:jmax] = l
        
        val_loss /= N
        val_acc = self.compute_accuracy(d0_all, d1_all, l_all)
        
        return val_loss, val_acc
    
    def validate_all(self):
        val_results = {}
        for k, v in self.val_datasets.items():
            val_loss, val_acc = self.validate(v)
            val_results[k+'/loss'] = val_loss.item()
            val_results[k+'/acc'] = val_acc
        return val_results
    
    def train(self, steps, save_every=5000, eval_every=5000):
        step=0
        ckpt_path=os.path.join(self.savedir, 'checkpoint.pt')
        if os.path.exists(ckpt_path):
            step = self.load(ckpt_path)
            print("Loaded checkpoint at step %s" % step)
        
        wandb.init(project='lpips')
        next_save = step + save_every
        next_eval = step + eval_every
        with tqdm.tqdm(total=steps) as pbar:
            while step < steps:
                for i, data in enumerate(self.train_dataset):
                    self.opt.zero_grad()
                    xref, x0, x1, l = data['ref'], data['p0'], data['p1'], data['judge']
                    xref, x0, x1, l = xref.to(self.device), x0.to(self.device), x1.to(self.device), l.to(self.device)
                    loss, acc = self.train_step(xref, x0, x1, l)
                    loss.backward()
                    self.opt.step()
                    self.clamp_weights()

                    logdict={
                        'train/loss': loss,
                        'train/acc': acc
                    }
                    
                    if step - self.batch_size < next_eval and step >= next_eval:
                        val_metrics = self.validate_all()
                        logdict.update(val_metrics)
                        next_eval += eval_every
                    
                    wandb.log(logdict, step)
                    
                    if step - self.batch_size < next_save and step >= next_save:
                        self.save(ckpt_path, next_save)
                        next_save += save_every
                        
                    step += self.batch_size
                    pbar.update(self.batch_size)
                    if step >= steps:
                        break
                        
        out_path=os.path.join(self.outdir, 'final.pt')
        self.save(out_path)
                    
            
                
import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run_name')
    parser.add_argument('bs', default=50, type=int)
    parser.add_argument('lr', default=1e-4, type=float)
    parser.add_argument('steps', default=250000, type=int)
    parser.add_argument('save_every', default=10000, type=int)
    parser.add_argument('eval_every', default=10000, type=int)
    args = parser.parse_args

    trainer = trainer = Trainer(args.bs, args.lr, '/checkpoint/kaselby/%s' % args.run_name, 'lpips_exps/%s'% args.run_name)
    trainer.train(args.steps, eval_every=args.eval_every, save_every=args.save_every)

