#import python libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import skimage as ski
import matplotlib.pyplot as plt
import numpy as np
import argparse

#import the functions from the other scripts
from HMDataLoading import HMDataset
from preprocess import *
from model import *
from train_eval import *
from plot import *



if __name__ == "__main__":
    
    # parse user defined inputs:
    parser = argparse.ArgumentParser("preprocess images")
    parser.add_argument("-i","--input_list", type=str, default='data/preprocessed_image_list.txt', help="the path of a list of image file paths to be loaded")
    parser.add_argument("-m","--model", type=str, default='autoencoder', help="the model architecture being used")
    parser.add_argument("-b","--batch_size", type=int, default=32, help="the batch size for the network training")
    parser.add_argument("-l","--learning_rate", type=float, default=0.001, help="the learning rate for the network training")
    parser.add_argument("-r","--train_test_ratio", type=float, default=0.8, help="the train test ratio for the network training")
    parser.add_argument("-e","--epoch", type=int, default=10, help="the epoch for the network training")
    parser.add_argument("-n","--newdata", action='store_true', help="if using a new split of data")
    
    args = parser.parse_args()
    print("--------------parameters used: -----------------")
    print("model:", args.model)
    print("batchsize:", args.batch_size)
    print("learning rate:", args.learning_rate)
    print("number of epochs:", args.epoch)
    print("--------------parameters used: -----------------")
    
    # load the dataset and construct the dataloader
    print("start loading the images and putting them into dataloaders")
    image_files = args.input_list
    batchSize = args.batch_size
    train_test_ratio = args.train_test_ratio
    
    if args.newdata:
        dataset = HMDataset(image_files)
        train_size, test_size = int(len(dataset)*train_test_ratio), len(dataset)-int(len(dataset)*train_test_ratio)
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
        torch.save(train_dataset, './torch_dataset2/train_dataset.pt')
        torch.save(val_dataset, './torch_dataset2/val_dataset.pt')
        
    else:
        train_dataset, val_dataset = torch.load('./torch_dataset/train_dataset.pt'), torch.load('./torch_dataset/val_dataset.pt')

    train_dataloader = DataLoader(train_dataset, batch_size=batchSize, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batchSize, shuffle=True)

    
    
    # define model and training methods
    print("start defining model and parameters for the training")
    if args.model == "autoencoder":
        model = Autoencoder()
        KL = False
    if args.model == "autoencoder_paper":
        model = Autoencoder_paper()
        KL = False
    if args.model == "autoencoder_mod":
        model = Autoencoder_mod()
        KL = False
    if args.model == "VAE":
        model = VAE()
        KL = True
    print("adding KL loss:", KL)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device using:", device)
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # train the model
    print("start training the model")
    mse_train_list, mse_val_list = [], []
    epoch = args.epoch
    for t in range(epoch):
        print(f"Epoch {t+1}\n-------------------------------")
        print("training process:")
        _ = Train(train_dataloader, model, criterion, optimizer, device, KL)
        print("evaluation process:")
        mse_train = Eval(train_dataloader, model, criterion, device, KL)
        mse_val = Eval(val_dataloader, model, criterion, device, KL)
        print(f"Training set: Avg MSE loss={mse_train}")
        print(f"Validation set: Avg MSE loss={mse_val}")
        mse_train_list.append(mse_train)
        mse_val_list.append(mse_val)
    torch.save(model.state_dict(), "trained_models2/" + str(args.model) + "_lr" + str(args.learning_rate) +"_bt" + str(args.batch_size)+   ".pt")
        
    print("Finish training!")
    
    
    # Plot the losses:
    print("Start plotting the loss values")
    FN = "lossplot_lr=" + str(args.learning_rate) +"bt=" + str(args.batch_size) + "_" + str(args.model)+ ".png"
    Plotloss(mse_train_list, mse_val_list, FN)

    