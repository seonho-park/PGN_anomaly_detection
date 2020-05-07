import torch
from torch import optim

import os
import numpy as np
from sklearn import metrics
import time

import utils
import dataset


def calculate_scores(mu, logvar):
    scores = -0.5*(1+logvar-mu.pow(2) - logvar.exp())
    scores = scores.sum(dim=1)
    return scores

def loss_function(mu, logvar):
    kl_div_loss = torch.sum(calculate_scores(mu, logvar))
    return kl_div_loss

def train(epoch, model, trainloader, optimizer, scheduler, logger, device):
    train_loss = 0.
    model.train() # train mode
    scheduler.step() # update optimizer lr
    
    for batch_idx, (inputs, _) in enumerate(trainloader):
        img_data = inputs.detach().cpu().numpy()[0].squeeze()
        inputs = inputs.to(device)
        optimizer.zero_grad()
        mu, logvar = model(inputs)
        loss = loss_function(mu,logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        print('  Training... Epoch: %4d | Iter: %4d/%4d | Mean Loss: %.4f '%(epoch, batch_idx+1, len(trainloader), train_loss/(batch_idx+1)), end = '\r')
    print('')
    logger.write('  Training... Epoch: %4d | Iter: %4d/%4d | Mean Loss: %.4f \n'%(epoch, batch_idx+1, len(trainloader), train_loss/(batch_idx+1)))


def pretrain(trainloader, datatype, device):
    import models
    ae_net = models.get_ae(datatype).to(device)
    optimizer = optim.Adam(ae_net.parameters(), lr=0.0001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[75], gamma=0.1)

    print('==> Start pretraining ..')   
    ae_net.train()
    for epoch in range(100):
        scheduler.step() # update optimizer lr
        train_loss = 0.
        for batch_idx, (inputs, _) in enumerate(trainloader):
            inputs = inputs.to(device)
            optimizer.zero_grad()
            outputs = ae_net(inputs)
            scores = torch.sum((outputs - inputs) ** 2, dim=tuple(range(1, outputs.dim())))
            recon_loss = torch.mean(scores) # reconstruction loss
            train_loss += recon_loss.item()
            recon_loss.backward()
            optimizer.step()

            print('  Pretraining... Epoch: %4d | Iter: %4d/%4d | Mean Loss: %.4f '%(epoch, batch_idx+1, len(trainloader), train_loss/(batch_idx+1)), end = '\r')
        print("")
    return ae_net


def test(model, testloader, T, device):
    test_loss = 0.
    scores_list = []
    scores_wo_variation_list = []
    targets_list = []

    model.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs = inputs.to(device)
            kl_dropout = []
            mu_dropout = []
            logvar_dropout = []
            for t in range(T):
                mu, logvar = model(inputs)
                kl_value = calculate_scores(mu, logvar)
                kl_dropout.append(kl_value.unsqueeze(dim=1))
                mu_dropout.append(mu.unsqueeze(dim=2))
                logvar_dropout.append(logvar.unsqueeze(dim=2))
           
            mu_dropout = torch.cat(mu_dropout, dim=2)
            logvar_dropout = torch.cat(logvar_dropout, dim=2)
            mu_mean = torch.mean(mu_dropout,dim=2)
            logvar_mean = torch.mean(logvar_dropout,dim=2)
            kl_value = calculate_scores(mu_mean,logvar_mean) # anomaly scores without taking into account variation
            scores_wo_variation_list.append(kl_value.cpu().numpy())

            scores = torch.cat(kl_dropout,dim=1).mean(dim=1) # anomaly scores considering variation
            scores_list.append(scores.cpu().numpy())
            
            targets_list.append(targets.cpu().numpy())
            print('  Test... Iter: %4d/%4d '%(batch_idx+1, len(testloader), ), end = '\r')
    print('')

    test_loss = test_loss/(batch_idx+1)
    targets = np.concatenate(targets_list)
    scores_total = np.concatenate(scores_list)
    auroc = metrics.roc_auc_score(targets, scores_total)

    scores_wo_variation_list_total = np.concatenate(scores_wo_variation_list)
    auroc_wo_variation = metrics.roc_auc_score(targets, scores_wo_variation_list_total)

    print('AUROC (proposed): %.4f, AUROC (without variation): %.4f'%(auroc,auroc_wo_variation))

    # calculate AUPR
    precision, recall, _ = metrics.precision_recall_curve(targets, scores_total)
    aupr = metrics.auc(recall, precision)
    
    return auroc, aupr, test_loss

def main(args):
    logger, result_dir, dir_name = utils.config_backup_get_log(args,__file__)
    device = utils.get_device()
    utils.set_seed(args.seed, device)

    trainloader = dataset.get_trainloader(args.data, args.dataroot, args.target, args.bstrain, args.nworkers)
    testloader = dataset.get_testloader(args.data, args.dataroot, args.target, args.bstest, args.nworkers)

    import models
    model = models.get_pgn_encoder(args.data, args.dropoutp).to(device)
   
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.999), weight_decay=1e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=0.1)   

    chpt_name = 'GPN_%s_target%s_seed%s.pth'%(args.data, str(args.target), str(args.seed))
    chpt_name = os.path.join("./chpt",chpt_name)

    print('==> Start training ..')   
    start = time.time()
    
    for epoch in range(args.maxepoch):
        train(epoch, model, trainloader, optimizer, scheduler, logger, device)

    auroc, aupr, _ = test(model, testloader, args.mcdropoutT, device)
    print('Epoch: %4d AUROC: %.6f AUPR: %.6f'%(epoch, auroc, aupr))
    logger.write('Epoch: %4d AUROC: %.6f AUPR: %.6f \n'%(epoch, auroc, aupr))
    state = {'model': model.state_dict(), 'auroc': auroc, 'epoch': epoch}
    torch.save(state, chpt_name)

    end = time.time()
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    print('AUROC... ', auroc)
    print("Elapsed Time: {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))
    logger.write("AUROC: %.8f\n"%(auroc))
    logger.write("Elapsed Time: {:0>2}:{:0>2}:{:05.2f}\n".format(int(hours),int(minutes),seconds))

if __name__ == '__main__':
    args = utils.process_args()
    main(args)
