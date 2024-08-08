import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

            
        
def Train(dataloader, model, criterion, optimizer, device, KL=False):
    totalsize = len(dataloader.dataset)
    model.train()
    totalLoss = []
    for batch_num, data in enumerate(dataloader):
        # format data
        X, _ = data
        X = X.to(device)
        X = X.float()
        # model outputs
        if KL:
            X_rec, X_latent, x_mu, x_logvar = model(X)
            loss = criterion(X_rec, X)
            loss_with_KL = criterion(X_rec, X) + KL_loss(x_mu, x_logvar)
            # back propagation
            optimizer.zero_grad()
            loss_with_KL.backward()
            optimizer.step()
                
        else:    
            X_rec, X_latent = model(X)
            loss = criterion(X_rec, X)
            # back propagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # record the loss
        totalLoss.append(loss.item())
        
        if (batch_num+1) % 5 == 0:
            loss = loss.item()
            num_trained = (batch_num+1) * len(X)
            print(f"Training set: Batch Avg loss= {loss:>7f}  [{num_trained:>5d}/{totalsize:>5d}]")
    final_avg_loss = sum(totalLoss)/len(totalLoss)
    #print(f"Training set: Avg MSE loss= {final_avg_loss}")
    return final_avg_loss
        


def Eval(dataloader, model, criterion, device, KL=False):
    totalsize = len(dataloader.dataset)
    model.eval()
    totalLoss = []
    with torch.no_grad():
        for data in dataloader:
            # format data 
            X, _ = data
            X = X.to(device)
            X = X.float()
            # model outputs
            if KL:
                X_rec, X_latent, _, _ = model(X)
            else:    
                X_rec, X_latent = model(X)
                
            loss = criterion(X_rec, X)
            # record the loss
            totalLoss.append(loss.item())
        avgLoss = sum(totalLoss)/len(totalLoss)
        #print(f"Validation set: Avg MSE loss={avgLoss}")
    return avgLoss

'''
# Calculate KL divergence for VAE
'''          
def KL_loss(mu, logvar, beta=0.001):
  # kl is the kl-divergence loss
  kl = 0.5 * torch.sum(- logvar + torch.exp(logvar) + mu * mu - 1)
  # beta is the regularization value
  return beta * kl  
    
    