import matplotlib.pyplot as plt

        
def Plotloss(mse_train_list, mse_val_list, fn="loss_plot.png"):
    """
    Plots the training and testing loss curves.

    Parameters:
    - mse_train: List of training loss values.
    - mse_val: List of testing loss values.
    - fn: file name to save the plot as
    """
    epochs = range(1, len(mse_val_list) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, mse_train_list, label='Training Loss', marker='o')
    plt.plot(epochs, mse_val_list, label='Testing Loss', marker='o')
    
    plt.title("loss curve")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.xticks(list(epochs))
    #plt.grid(True)
    
    plt.savefig(fn)    