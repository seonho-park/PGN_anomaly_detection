import torch
from torch import optim
from torch.autograd import Variable

import os
import numpy as np
from sklearn import metrics
import time
import itertools

import utils
import dataset


def train(epoch, adversarial_loss, pixelwise_loss, encoder, decoder, discriminator, trainloader, optimizer_G, optimizer_D, scheduler_G, scheduler_D, logger, device):
    train_G_loss = 0.
    train_D_loss = 0.

    encoder.train() # train mode
    decoder.train() # train mode
    discriminator.train() # train mode

    scheduler_G.step() # update optimizer lr
    scheduler_D.step() # update optimizer lr
    
    for batch_idx, (inputs, _) in enumerate(trainloader):
        inputs = inputs.to(device)
        valid = Variable(torch.FloatTensor(inputs.size(0), 1).fill_(1.0).to(device), requires_grad=False)
        fake = Variable(torch.FloatTensor(inputs.size(0), 1).fill_(0.0).to(device), requires_grad=False)

        # Train generator
        optimizer_G.zero_grad()
        encoded_img = encoder(inputs)
        decoded_img = decoder(encoded_img)

        g_loss = 0.001 * adversarial_loss(discriminator(encoded_img), valid) + 0.999 * pixelwise_loss(decoded_img, inputs)
        g_loss.backward()
        train_G_loss += g_loss.item()
        optimizer_G.step()

        # Train discriminator
        optimizer_D.zero_grad()
        z = torch.randn_like(encoded_img)
        
        real_loss = adversarial_loss(discriminator(z), valid)
        fake_loss = adversarial_loss(discriminator(encoded_img.detach()), fake)
        d_loss = 0.5 * (real_loss + fake_loss)
        d_loss.backward()
        train_D_loss += d_loss.item()
        optimizer_D.step()

        print('  Training... Epoch: %4d | Iter: %4d/%4d | Mean G Loss: %.4f | Mean D Loss: %.4f '%(epoch, batch_idx+1, len(trainloader), train_G_loss/(batch_idx+1), train_D_loss/(batch_idx+1)), end = '\r')
    print('')
    logger.write('  Training... Epoch: %4d | Iter: %4d/%4d | Mean G Loss: %.4f | Mean D Loss: %.4f \n'%(epoch, batch_idx+1, len(trainloader), train_G_loss/(batch_idx+1), train_D_loss/(batch_idx+1)))


def test(encoder, decoder, testloader, device):
    test_loss = 0.
    scores_list = []
    targets_list = []

    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs = inputs.to(device)
            encoded_img = encoder(inputs)
            decoded_img = decoder(encoded_img)
            scores = torch.sum((decoded_img - inputs) ** 2, dim=tuple(range(1, decoded_img.dim())))
            scores_list.append(scores.cpu().numpy())
            targets_list.append(targets.cpu().numpy())
            print('  Test... Iter: %4d/%4d '%(batch_idx+1, len(testloader)), end = '\r')
    print('')

    test_loss = test_loss/(batch_idx+1)
    scores = np.concatenate(scores_list)
    targets = np.concatenate(targets_list)
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
    encoder, decoder, discriminator = models.get_aae(args.data)
    encoder.to(device)
    decoder.to(device)
    discriminator.to(device)

    # Use binary cross-entropy loss
    adversarial_loss = torch.nn.BCELoss().to(device)
    pixelwise_loss = torch.nn.L1Loss().to(device)

    optimizer_G = torch.optim.Adam(itertools.chain(encoder.parameters(), decoder.parameters()), lr=args.lr, betas=(0.5, 0.999), weight_decay=1e-4)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999), weight_decay=1e-4)

    scheduler_G = optim.lr_scheduler.MultiStepLR(optimizer_G, milestones=args.milestones, gamma=0.1)   
    scheduler_D = optim.lr_scheduler.MultiStepLR(optimizer_D, milestones=args.milestones, gamma=0.1)   

    chpt_name = 'AAE_%s_target%s_seed%s.pth'%(args.data, str(args.target), str(args.seed))
    chpt_name = os.path.join("./chpt",chpt_name)

    print('==> Start training ..')   
    best_auroc = 0.
    start = time.time()
    for epoch in range(args.maxepoch):
        train(epoch, adversarial_loss, pixelwise_loss, encoder, decoder, discriminator, trainloader, optimizer_G, optimizer_D, scheduler_G, scheduler_D, logger, device)

    auroc, aupr, _ = test(encoder, decoder, testloader, device)
    print('Epoch: %4d AUROC: %.4f AUPR: %.4f'%(epoch, auroc, aupr))
    logger.write('Epoch: %4d AUROC: %.4f AUPR: %.4f \n'%(epoch, auroc, aupr))
    state = {
        'encoder': encoder.state_dict(), 
        'decoder': decoder.state_dict(), 
        'discriminator': discriminator.state_dict(), 
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
    main(args)
