import torch
from torch import optim
import torch.nn as nn

import os
import numpy as np
from sklearn import metrics
import argparse
import time
import csv

import utils
import dataset
import losses

class Distribution(object) :
    def sample(self) :
        raise NotImplementedError

    def log_prob(self,values) :
        raise NotImplementedError

class Bernoulli(Distribution):
    def __init__(self, probs):
        self.probs = probs

    def sample(self):
        return torch.bernoulli(self.probs)

    def log_prob(self,values):
        probs = self.probs.clamp(0.00001,0.99999)
        log_pmf = ( torch.stack( [1.-probs, probs] ) ).log()
        dum = values.unsqueeze(0).long()
        return log_pmf.gather( 0, dum ).squeeze(0)

def loss_function(recon_list, x, mu, logvar, beta):
    recon_loss = 0.
    D = 0.
    for recon_x in recon_list:
        D_batch = Bernoulli( recon_x.view((recon_x.size(0), -1)) ).log_prob( x.view(x.size(0),-1) ).sum(dim=1)
        D -= torch.mean( D_batch )
    D /= len(recon_list)
    R = 0.5 * torch.mean( torch.sum( (mu.pow(2) + logvar.exp() - logvar -1), dim=1) )
    return D+beta*R, D, R


def train(epoch, model, trainloader, optimizer, scheduler, logger, device, beta):
    train_loss = 0.
    model.train() # train mode
    distortion = 0.
    rate = 0.

    scheduler.step() # update optimizer lr
    
    for batch_idx, (inputs, _) in enumerate(trainloader):
        inputs = inputs.to(device)

        optimizer.zero_grad()
        recon_list, mu, logvar = model(inputs)
        loss, d, r = loss_function(recon_list, inputs, mu, logvar, beta)
        loss.backward()
        train_loss += loss.item()
        distortion += d.item()
        rate += r.item()
        optimizer.step()

    distortion = distortion/(batch_idx+1)
    rate = rate/(batch_idx+1)
    train_loss = train_loss/(batch_idx+1)
    return train_loss, distortion, rate

def test(model, trainloader, logger, device, beta):
    train_loss = 0.
    model.eval()
    distortion = 0.
    rate = 0.
    for batch_idx, (inputs, _) in enumerate(trainloader):
        inputs = inputs.to(device)
        recon_list, mu, logvar = model(inputs)
        loss, d, r = loss_function(recon_list, inputs, mu, logvar, beta)
        train_loss += loss.item()
        distortion += d.item()
        rate += r.item()
    distortion = distortion/(batch_idx+1)
    rate = rate/(batch_idx+1)
    train_loss = train_loss/(batch_idx+1)
    return train_loss, distortion, rate

def main(args):
    logger, result_dir, dir_name = utils.config_backup_get_log(args,__file__)
    csv_name = os.path.join(result_dir,'rate_distortion.csv')
    file = open(csv_name, 'w', newline ='')
    csvwriter = csv.writer(file)
    csvwriter.writerow(['beta', args.beta])

    device = utils.get_device()
    utils.set_seed(args.seed, device)

    trainloader = dataset.get_trainloader(args.data, args.dataroot, args.type, args.target, args.bstrain, args.nworkers)
    
    import models
    model = models.get_vae(args.data,L=10).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.999), weight_decay=1e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=0.1)   

    chpt_name = 'betaVAE_%s_target%s_seed%s.pth'%(args.data, str(args.target), str(args.seed))
    chpt_name = os.path.join("./chpt",chpt_name)

    print('==> Start training ..')   
    start = time.time()
    loss, distortion, rate = test(model, trainloader, logger, device, args.beta)
    csvwriter.writerow([-1,distortion, rate, distortion+rate])
    print('EPOCH %3d LOSS: %.4F, DISTORTION: %.4f, RATE: %.4f, D+R: %.4f'%(-1, loss, distortion, rate, distortion+rate))
    for epoch in range(args.maxepoch):
        loss, distortion, rate = train(epoch, model, trainloader, optimizer, scheduler, logger, device, args.beta)
        csvwriter.writerow([epoch,distortion, rate, distortion+rate])
        print('EPOCH %3d LOSS: %.4F, DISTORTION: %.4f, RATE: %.4f, D+R: %.4f'%(epoch, loss, distortion, rate, distortion+rate))

    
    file.close()
    end = time.time()
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("Elapsed Time: {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))
    logger.write("Elapsed Time: {:0>2}:{:0>2}:{:05.2f}\n".format(int(hours),int(minutes),seconds))
    
    if args.batchout:
        f = open('temp_result.txt', 'w', newline ='')
        f.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1003, help='random seed')
    parser.add_argument('--data', type=str, default='mnist', help='data type, mnist|cifar10|mvtec|bmnist')
    parser.add_argument('--dataroot', type=str, default='/home/sean/data', help='data path')
    parser.add_argument('--target', type=int, default=0, help='target integer in mnist dataset')
    parser.add_argument('--type', type=str, default='whole', help='AD type, unimodal|multimodal|whole') #multimodal is deprecated
    parser.add_argument('--bstrain', type=int, default=1000, help='batch size for training')
    parser.add_argument('--bstest', type=int, default=200, help='batch size for testing')
    parser.add_argument('--nworkers', type=int, default=8, help='the number of workers used in DataLoader')
    parser.add_argument('--maxepoch', type=int, default=100, help='the number of epoches')
    parser.add_argument('--dropoutp', type=float, default=0.5, help='dropout probability')
    parser.add_argument('--beta', type=float, default=5., help='beta of beta-VAE')
    parser.add_argument('--pretrain', type=bool, default=False, help='pretrain')
    parser.add_argument('--mcdropoutT', type=int, default=20, help='the number of mc samplings')
    parser.add_argument('--suffix', type=str, default='test', help='suffix of result directory')
    parser.add_argument('--batchout', action='store_true')
    args = parser.parse_args()
    args.lr = 0.001
    args.milestones = [500]
    args.maxepoch = 101
    main(args)
