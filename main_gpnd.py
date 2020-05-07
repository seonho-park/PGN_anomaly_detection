import torch
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F
from torch.autograd.gradcheck import zero_gradients

import os
import numpy as np
import scipy
from sklearn import metrics
import argparse
import time

import utils
import dataset


def compute_jacobian(x, z, device):
    assert x.requires_grad
    num_classes = z.size()[1]
    jacobian = torch.zeros(num_classes, *x.size(), device = device)
    grad_output = torch.zeros(*z.size(), device = device)
    for i in range(num_classes):
        zero_gradients(x)
        grad_output.zero_()
        grad_output[:, i] = 1
        z.backward(grad_output, retain_graph=True)
        jacobian[i] = x.grad.data
    return torch.transpose(jacobian, dim0=0, dim1=1)


def train(epoch, encoder, generator, discriminator, discriminator_z, trainloader, optimizer_G, optimizer_D, optimizer_E, optimizer_GE, optimizer_ZD, schedulers, logger, device):
    train_G_loss = 0.
    train_D_loss = 0.
    train_E_loss = 0.
    train_GE_loss = 0.
    train_ZD_loss = 0.
    
    encoder.train() # train mode
    generator.train() # train mode
    discriminator.train() # train mode
    discriminator_z.train() # train mode

    scheduler_G, scheduler_D, scheduler_E, scheduler_GE, scheduler_ZD = schedulers
    scheduler_G.step()
    scheduler_D.step()
    scheduler_E.step()
    scheduler_GE.step()
    scheduler_ZD.step()

    for batch_idx, (inputs, _) in enumerate(trainloader):
        inputs = inputs.to(device)
        valid = torch.ones(inputs.size(0), device=device)
        fake = torch.zeros(inputs.size(0), device=device)

        ################################################

        discriminator.zero_grad()
        D_result = discriminator(inputs).squeeze()
        D_real_loss = F.binary_cross_entropy(D_result, valid)

        z = Variable(torch.randn((inputs.size(0), encoder.rep_dim)).to(device))
        x_fake = generator(z).detach()
        D_result = discriminator(x_fake).squeeze()
        D_fake_loss = F.binary_cross_entropy(D_result, fake)
        D_train_loss = D_real_loss + D_fake_loss
        D_train_loss.backward()
        optimizer_D.step()
        train_D_loss += D_train_loss.item()
        
        ################################################

        generator.zero_grad()
        z = torch.randn((inputs.size(0), encoder.rep_dim)).to(device)
        x_fake = generator(z)
        D_result = discriminator(x_fake).squeeze()
        G_train_loss = F.binary_cross_entropy(D_result, valid)
        G_train_loss.backward()
        optimizer_G.step()
        train_G_loss += G_train_loss.item()

        ################################################
        discriminator_z.zero_grad()
        z = Variable(torch.randn((inputs.size(0), encoder.rep_dim)).to(device))

        result_zd = discriminator_z(z).squeeze()
        real_loss_zd = F.binary_cross_entropy(result_zd, valid)
        z = encoder(inputs).squeeze().detach()
        result_zd = discriminator_z(z).squeeze()
        fake_loss_zd = F.binary_cross_entropy(result_zd, fake)
        train_loss_zd = real_loss_zd + fake_loss_zd
        train_loss_zd.backward()
        optimizer_ZD.step()
        train_ZD_loss += train_loss_zd.item()

        ################################################

        encoder.zero_grad()
        generator.zero_grad()

        z = encoder(inputs)
        x_d = generator(z)

        result_zd = discriminator_z(z.squeeze()).squeeze()

        E_loss = F.binary_cross_entropy(result_zd, valid) * 2.0
        recon_loss = F.binary_cross_entropy(x_d, inputs)
        (recon_loss + E_loss).backward()
        optimizer_GE.step()

        train_GE_loss += recon_loss.item()
        train_E_loss += E_loss.item()

        print('  Training... Epoch: %4d | Iter: %4d/%4d | Gloss: %.3f | Dloss: %.3f | ZDloss: %.3f | GEloss: %.3f | Eloss: %.3f'%(epoch, batch_idx+1, len(trainloader), train_G_loss/(batch_idx+1), train_D_loss/(batch_idx+1), train_ZD_loss/(batch_idx+1), train_GE_loss/(batch_idx+1), train_E_loss/(batch_idx+1)), end = '\r')
    print('')
    logger.write('  Training... Epoch: %4d | Iter: %4d/%4d | Gloss: %.3f | Dloss: %.3f | ZDloss: %.3f | GEloss: %.3f | Eloss: %.3f \n'%(epoch, batch_idx+1, len(trainloader), train_G_loss/(batch_idx+1), train_D_loss/(batch_idx+1), train_ZD_loss/(batch_idx+1), train_GE_loss/(batch_idx+1), train_E_loss/(batch_idx+1)))


def test(encoder, generator, testloader, device):
    test_loss = 0.
    targets_list = []

    encoder.eval()
    generator.eval()

    rlist = []
    zlist = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs = inputs.to(device)

            encoded_img = encoder(inputs)
            decoded_img = generator(encoded_img)

            distance = torch.sum((decoded_img - inputs) ** 2, dim=tuple(range(1, decoded_img.dim())))
            rlist.append(distance.cpu().numpy())
            zlist.append(encoded_img.cpu().detach().numpy())

            # scores_list.append(scores.cpu().numpy())
            targets_list.append(targets.cpu().numpy())
            print('  Test... Iter: %4d/%4d '%(batch_idx+1, len(testloader)), end = '\r')
    print('')

    test_loss = test_loss/(batch_idx+1)
    rlist = np.concatenate(rlist)
    zlist = np.concatenate(zlist)
    counts, bin_edges = np.histogram(rlist, bins=30, normed=True)
    def r_pdf(x, bins, count):
        if x < bins[0]:
            return max(count[0], 1e-308)
        if x >= bins[-1]:
            return max(count[-1], 1e-308)
        id = np.digitize(x, bins) - 1
        return max(count[id], 1e-308)
    
    gennorm_param = np.zeros([3, encoder.rep_dim])
    for i in range(encoder.rep_dim):
        betta, loc, scale = scipy.stats.gennorm.fit(zlist[:, i])
        gennorm_param[0, i] = betta
        gennorm_param[1, i] = loc
        gennorm_param[2, i] = scale

    scores_list = []
    targets_list = []
    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs = inputs.to(device)
        inputs = Variable(inputs, requires_grad = True)
        encoded_img = encoder(inputs)
        decoded_img = generator(encoded_img)
        
        J = compute_jacobian(inputs, encoded_img, device).cpu().numpy()
        encoded_img = encoded_img.cpu().detach().numpy()
        decoded_img = decoded_img.squeeze().cpu().detach().numpy()
        inputs = inputs.squeeze().cpu().detach().numpy()

        for i in range(inputs.shape[0]):
            u, s, vh = np.linalg.svd(J[i, :, :], full_matrices=False)
            logD = np.sum(np.log(np.abs(s))) # | \mathrm{det} S^{-1} |
            p = scipy.stats.gennorm.pdf(encoded_img[i], gennorm_param[0, :], gennorm_param[1, :], gennorm_param[2, :])
            logPz = np.sum(np.log(p))
            if not np.isfinite(logPz):
                logPz = -1000
            distance = np.sum(np.power(inputs[i].flatten() - decoded_img[i].flatten(), 2))
            logPe = np.log(r_pdf(distance, bin_edges, counts)) # p_{\|W^{\perp}\|} (\|w^{\perp}\|)
            logPe -= np.log(distance) * (np.prod(inputs.shape[1:]) - encoder.rep_dim) # \| w^{\perp} \|}^{m-n}
            P = logD + logPz + logPe
            scores_list.append(P)
            targets_list.append(targets[i].item())

    scores = np.asarray(scores_list)
    targets = 1.-np.asarray(targets_list) 
    auroc = metrics.roc_auc_score(targets, scores)

    precision, recall, _ = metrics.precision_recall_curve(targets, scores)
    aupr = metrics.auc(recall, precision)

    return auroc, aupr, test_loss


def main(args):
    logger, result_dir, dir_name = utils.config_backup_get_log(args,__file__)
    device = utils.get_device()
    utils.set_seed(args.seed, device)

    trainloader = dataset.get_trainloader(args.data, args.dataroot, args.target, args.bstrain, args.nworkers)
    testloader = dataset.get_testloader(args.data, args.dataroot, args.target, args.bstest, args.nworkers)
    
    import models
    encoder, generator, discriminator, discriminator_z = models.get_gpnd(args.data)
    encoder.to(device)
    generator.to(device)
    discriminator.to(device)
    discriminator_z.to(device)

    optimizer_G = optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizer_E = optim.Adam(encoder.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizer_GE = optim.Adam(list(encoder.parameters()) + list(generator.parameters()), lr=args.lr, betas=(0.5, 0.999))
    optimizer_ZD = optim.Adam(discriminator_z.parameters(), lr=args.lr, betas=(0.5, 0.999))

    scheduler_G = optim.lr_scheduler.MultiStepLR(optimizer_G, milestones=args.milestones, gamma=args.gamma)   
    scheduler_D = optim.lr_scheduler.MultiStepLR(optimizer_D, milestones=args.milestones, gamma=args.gamma)   
    scheduler_E = optim.lr_scheduler.MultiStepLR(optimizer_E, milestones=args.milestones, gamma=args.gamma)   
    scheduler_GE = optim.lr_scheduler.MultiStepLR(optimizer_GE, milestones=args.milestones, gamma=args.gamma)   
    scheduler_ZD = optim.lr_scheduler.MultiStepLR(optimizer_ZD, milestones=args.milestones, gamma=args.gamma)   
    schedulers = (scheduler_G, scheduler_D, scheduler_E, scheduler_GE, scheduler_ZD)
    chpt_name = 'GPND_%s_target%s_seed%s.pth'%(args.data, str(args.target), str(args.seed))
    chpt_name = os.path.join("./chpt",chpt_name)

    print('==> Start training ..')   
    start = time.time()
    for epoch in range(args.maxepoch):
        train(epoch, encoder, generator, discriminator, discriminator_z, trainloader, optimizer_G, optimizer_D, optimizer_E, optimizer_GE, optimizer_ZD, schedulers, logger, device)
        if epoch > 79 and epoch % 20==0:
            auroc, aupr, _ = test(encoder, generator, testloader, device)
            print(auroc)

    auroc, aupr, _ = test(encoder, generator, testloader, device)
    print('Epoch: %4d AUROC: %.4f AUPR: %.4f'%(epoch, auroc, aupr))
    state = {
        'encoder': encoder.state_dict(), 
        'generator': generator.state_dict(), 
        'discriminator': discriminator.state_dict(), 
        'discriminator_z': discriminator_z.state_dict(), 
        'auroc': auroc, 
        'epoch': epoch}
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
    args.lr = 0.002
    args.milestones = [30,60]
    args.gamma = 0.25
    args.bstrain = 128
    args.maxepoch = 80
    main(args)
