import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms



# Define an VAE model
#TODO: implement KL divergence
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            # Input shape: (batch, 3, 64, 64)
            # First Conv Layer
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        
            # Second Conv Layer
            nn.Conv2d(16, 8, kernel_size=3, stride=2, padding=1),
            nn.Sigmoid(),
            
            # Flatten the image
            nn.Flatten(), #shape: (batch, 8*16*16) 
        )
        self.encoder_mu = nn.Linear(8*16*16, 8*16*16)
        self.encoder_logvar = nn.Linear(8*16*16, 8*16*16)
        
        self.decoder = nn.Sequential(
            # Reshape the image
            nn.Unflatten(1, (8,16,16)),
            # First DeConv Layer
            nn.ConvTranspose2d(8, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            # Second DeConv Layer
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
            # Output shape: (batch, 3, 64, 64)   
        )
        
    def Sample(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar * 0.5) #standard deviation
            eps = torch.empty_like(std).normal_() #error term   ##same shape with std, but is ~ N(0,1)
            z = mu + (std * eps)
            return z
        else:
            return mu       
    
    def forward(self, x):
        x_encode = self.encoder(x)
        x_mu, x_logvar = self.encoder_mu(x_encode), self.encoder_logvar(x_encode)
        x_latent = self.Sample(x_mu, x_logvar)
        x_reconstruct = self.decoder(x_latent)
        return x_reconstruct, x_latent, x_mu, x_logvar






# Define the autoencoder model

#Conv2d: (in_channels, out_channels, kernel_size, stride=1, padding=0)
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            # Input shape: (batch, 3, 64, 64)
            # First Conv Layer
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Second Conv Layer
            nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), #shape: (batch, 8, 16, 16)
            # Flatten the image
            nn.Flatten(), #shape: (batch, 8*16*16)
            # Dense Layer to reduce it to two dimensions
            nn.Linear(8*16*16, 2)
            
        )
        self.decoder = nn.Sequential(
            # Dense Layer to recover it back to 8*16*16
            nn.Linear(2, 8*16*16),
            # Reshape the image
            nn.Unflatten(1, (8,16,16)),
                    
            # First DeConv Layer
            nn.ConvTranspose2d(8, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            # Second DeConv Layer
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
            # Output shape: (batch, 3, 64, 64)
            
        )
    
    def forward(self, x):
        x_latent = self.encoder(x)
        x_reconstruct = self.decoder(x_latent)
        return x_reconstruct, x_latent


#Conv2d: (in_channels, out_channels, kernel_size, stride=1, padding=0)
class Autoencoder_paper(nn.Module):
    # Reference: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8622369/
    def __init__(self):
        super(Autoencoder_paper, self).__init__()
        self.encoder = nn.Sequential(
            # Input shape: (batch, 3, 64, 64)
            # First Conv Layer
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
    
            # Second Conv Layer
            nn.Conv2d(16, 8, kernel_size=3, stride=2, padding=1),
            nn.Sigmoid()
            # Flatten the image
            #nn.Flatten(), #shape: (batch, 8*16*16)
            # Dense Layer to reduce it to two dimensions
            #nn.Linear(8*16*16, 2)
            
        )
        self.decoder = nn.Sequential(
            # Dense Layer to recover it back to 8*16*16
            #nn.Linear(2, 8*16*16),
            # Reshape the image
            #nn.Unflatten(1, (8,16,16)),
                    
            # First DeConv Layer
            nn.ConvTranspose2d(8, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            # Second DeConv Layer
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
            # Output shape: (batch, 3, 64, 64)
            
        )
    
    def forward(self, x):
        x_latent = self.encoder(x)
        x_reconstruct = self.decoder(x_latent)
        return x_reconstruct, x_latent


class Autoencoder_mod(nn.Module):
    # Reference: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8622369/
    def __init__(self):
        super(Autoencoder_mod, self).__init__()
        self.encoder = nn.Sequential(
            # Input shape: (batch, 3, 64, 64)
            # First Conv Layer
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
    
            # Second Conv Layer
            nn.Conv2d(16, 8, kernel_size=3, stride=2, padding=1),
            nn.Sigmoid(),
            # Flatten the image
            nn.Flatten(), #shape: (batch, 8*16*16)
            # Dense Layer to reduce it to two dimensions
            nn.Linear(8*16*16, 4*16*16)
            
        )
        self.decoder = nn.Sequential(
            # Dense Layer to recover it back to 8*16*16
            nn.Linear(4*16*16, 8*16*16),
            # Reshape the image
            nn.Unflatten(1, (8,16,16)),
                    
            # First DeConv Layer
            nn.ConvTranspose2d(8, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            # Second DeConv Layer
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
            # Output shape: (batch, 3, 64, 64)
            
        )
    
    def forward(self, x):
        x_latent = self.encoder(x)
        x_reconstruct = self.decoder(x_latent)
        return x_reconstruct, x_latent





if __name__ == "__main__":
    
    #testing the model strucutre
    val_dataset = torch.load('./torch_dataset/val_dataset.pt')
    val_dataloader = DataLoader(val_dataset, batch_size=3, shuffle=False)
    
    #defining model
    model = VAE()
    
    #input one image
    for data in val_dataloader:
        # format data 
        X, imgPATH = data
        X = X.float()
        # model outputs
        X_rec, X_latent = model(X)
        print("shape of X is:", X.size())
        print("shape of X_latent is:", X_latent.size())
        print("shape of X_rec is:", X_rec.size())
        break