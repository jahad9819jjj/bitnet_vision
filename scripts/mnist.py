import sys
import os
import pathlib
script_path = pathlib.Path(os.path.abspath(__file__))
sys.path.append(str(script_path.parent.parent))
import torch
import torch.nn as nn
from torchvision import datasets, transforms
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

import tqdm

def save_model(model, filename):
  """Saves the PyTorch model to the specified file.

  Args:
      model (torch.nn.Module): The model to be saved.
      filename (str): The filename to use for saving the model.
  """

  # Save only the model state dictionary for efficiency and compatibility.
  torch.save(model.state_dict(), filename)
  print(f"Model saved successfully to '{filename}'.")
  
  
if __name__ == "__main__":
    if torch.cuda.is_available():
        print("Cuda is available")
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        print("MPS is available")
        device = torch.device("mps")
    else:
        print("Cuda is not available")
        device = torch.device("cpu")
    print(device)
    torch.manual_seed(0)
    np.random.seed(0)
    BATCH_SIZE = 64
    LR = 5e-5
    NUM_EPOCHES = 25
    mean, std = (0.5,), (0.5,)

    transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean, std)
                              ])
    
    trainset = datasets.MNIST('../data/MNIST/', download=True, train=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)

    testset = datasets.MNIST('../data/MNIST/', download=True, train=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)
    
    from model.bitvit import BitNetViT, VisionEncoder
    image_size = 28
    channel_size = 1
    patch_size = 7
    embed_size = 512
    num_heads = 8
    classes = 10
    num_layers = 3
    hidden_size = 256
    dropout = 0.2

    model = BitNetViT(image_size, channel_size, patch_size, embed_size, num_heads, classes, num_layers, hidden_size, dropout=dropout).to(device)

    for img, label in trainloader:
        img = img.to(device)
        label = label.to(device)
        
        print("Input Image Dimensions: {}".format(img.size()))
        print("Label Dimensions: {}".format(label.size()))
        print("-"*100)
        
        out = model(img)
        
        print("Output Dimensions: {}".format(out.size()))
        break
    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LR)
    
    loss_hist = {}
    loss_hist["train accuracy"] = []
    loss_hist["train loss"] = []

    for epoch in tqdm.tqdm(range(1, NUM_EPOCHES+1), mininterval=10.0, desc="training"):
        model.train()
        
        epoch_train_loss = 0
            
        y_true_train = []
        y_pred_train = []
            
        for batch_idx, (img, labels) in enumerate(trainloader):
            img = img.to(device)
            labels = labels.to(device)
            
            preds = model(img)
            
            loss = criterion(preds, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            y_pred_train.extend(preds.detach().argmax(dim=-1).tolist())
            y_true_train.extend(labels.detach().tolist())
                
            epoch_train_loss += loss.item()
        
        loss_hist["train loss"].append(epoch_train_loss)
        
        total_correct = len([True for x, y in zip(y_pred_train, y_true_train) if x==y])
        total = len(y_pred_train)
        accuracy = total_correct * 100 / total
        
        loss_hist["train accuracy"].append(accuracy)
        
        print("-------------------------------------------------")
        print("Epoch: {} Train mean loss: {:.8f}".format(epoch, epoch_train_loss))
        print("       Train Accuracy%: ", accuracy, "==", total_correct, "/", total)
        print("-------------------------------------------------")
        plt.plot(loss_hist["train accuracy"])
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.savefig("train_accuracy.png")

        plt.plot(loss_hist["train loss"])
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.savefig("train_loss.png")
        
        with torch.no_grad():
            model.eval()
            
            y_true_test = []
            y_pred_test = []
            
            for batch_idx, (img, labels) in enumerate(testloader):
                img = img.to(device)
                label = label.to(device)
            
                preds = model(img)
                
                y_pred_test.extend(preds.detach().argmax(dim=-1).tolist())
                y_true_test.extend(labels.detach().tolist())
                
            total_correct = len([True for x, y in zip(y_pred_test, y_true_test) if x==y])
            total = len(y_pred_test)
            accuracy = total_correct * 100 / total
            
            print("Test Accuracy%: ", accuracy, "==", total_correct, "/", total)
        save_model(model, 'mnist_model.pth')