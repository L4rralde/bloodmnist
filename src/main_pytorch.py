import os

import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms

from medmnist import BloodMNIST
from utils import GIT_ROOT


def one_hot(y):
    onehot_y = torch.zeros(8, dtype=torch.float)
    onehot_y.scatter_(0, torch.tensor(y), value=1)
    return onehot_y


def train(model):
    DATASETS_PATH = f"{GIT_ROOT}/datasets/"

    my_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    target_transforms = transforms.Lambda(one_hot)

    train_data = BloodMNIST(
        "train",
        size=224,
        root=DATASETS_PATH,
        as_rgb=True,
        transform=my_transforms,
        target_transform=target_transforms
    )

    val_data = BloodMNIST(
        "val",
        size=224,
        root=DATASETS_PATH,
        as_rgb=True,
        transform=my_transforms,
        target_transform=target_transforms
    )

    test_data = BloodMNIST(
        "test",
        size=224,
        root=DATASETS_PATH,
        as_rgb=True,
        transform=my_transforms,
        target_transform=target_transforms
    )

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    batch_size = 64
    train_data_loader = DataLoader(train_data, batch_size = batch_size, shuffle=True)
    val_data_loader = DataLoader(val_data, batch_size = batch_size)
    test_data_loader = DataLoader(test_data, batch_size = batch_size)

    model = model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = 0.001)

    val_losses = []
    best_val_loss = float('inf')

    epochs = 10
    for epoch in range(epochs):
        train_loss = 0.0
        model.train()# Sets the model in training mode.
                        #src codde:
                        #self.training = mode
                        #for module in self.children():
                        #    module.train(mode)
                        #return self
                        #dropout and batchnorm behaved differently on training
        for batch in train_data_loader:
            optimizer.zero_grad() #explicitly sets the gradients to zero
            #This accumulating behavior is convenient while training RNNs
            #or when we want to compute the gradient of the loss summed 
            #over multiple mini-batches. 
            #So, the default action has been set to accumulate 
            #(i.e. sum) the gradients on every loss.backward() call
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            y_hat = model(x) #Prediction y_hat = model.forward(x)
            #print(y, y_hat)
            loss = loss_fn(y_hat, y)
            loss.backward() #computes dloss/dx for every parameter 
                            #x which has requires_grad=True
            optimizer.step() #updates the value of x using the gradient x.grad
            train_loss += loss.data.item()
        train_loss /= len(train_data_loader)

        model.eval() #Unsets training mode.
                        # src code:
                        #return self.train(False)
        val_loss = 0.0
        correct = 0
        with torch.no_grad():
            for batch in val_data_loader:
                x, y = batch
                x = x.to(device)
                y = y.to(device)
                y_hat = model(x)
                val_loss += loss_fn(y_hat, y).item()
                correct += (y_hat.argmax(1) == y.argmax(1)).type(torch.float).sum().item()
            val_loss /= len(val_data_loader)
            correct /= len(val_data_loader.dataset)
            print(f"{epoch+1}/{epochs}. Accuracy:{correct}, Train loss:{train_loss}, Val loss:{val_loss}")
            val_losses.append(val_loss)
            if(val_loss < best_val_loss):
                best_val_loss = val_loss
                model_path = f"models/{type(model).__name__}_{val_loss}"
                print(f"Saving model at: {model_path}")
                torch.save(model.state_dict(), model_path)
    print(val_losses)


def get_default_model():
    model = models.vgg16(weights='DEFAULT')
    model.classifier = nn.Sequential(
        nn.Linear(25088, 4096),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(4096, 256),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(256, 8),
    )
    return model

if __name__ == '__main__':
    if not os.path.exists(f"{GIT_ROOT}/models"):
        os.makedirs(f"{GIT_ROOT}/models")

    model = get_default_model()

    for name, param in model.named_parameters():
        if not "classifier" in name:
            param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True

    print(model)

    train(model)
