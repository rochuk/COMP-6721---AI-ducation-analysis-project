import torch
import numpy as np

from torch import optim, nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as td
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split




# Function to load dataset
def dataset_loader(batch_size, shuffle_test=False):
    # Normalization values for CIFAR10 dataset
    normalize = transforms.Normalize(mean=[0.5],
    std=[0.5])
    # Loading training dataset with data augmentation techniques
    dataset = datasets.ImageFolder(root='/Users/roshinichukkapalli/Documents/AI/AIdatasets/TrainDataset',
    transform=transforms.Compose([
     transforms.Resize((224, 224)),   
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(224, 4),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    normalize
    ]))
    train_dataset, test_dataset = train_test_split(dataset, test_size=0.3, random_state=42)
    val_dataset, test_dataset = train_test_split(test_dataset, test_size=0.5, random_state=42)

    # Creating data loaders for training, validation, and testing
    train_loader = td.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = td.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    test_loader = td.DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle_test, pin_memory=True)

    return train_loader, val_loader, test_loader

# Creating CNN: main architecture
class MultiLayerFCNet(nn.Module):
    def __init__(self,input_size, hidden_size, output_size):
        super().__init__()
        self.layer1=nn.Conv2d(1,32,3,padding=1,stride=1)
        self.B1 = nn.BatchNorm2d(32)
        self.layer2 = nn.Conv2d(32, 32, 3, padding=1, stride=1)
        self.B2 = nn.BatchNorm2d(32)
        self.Maxpool=nn.MaxPool2d(2)
        self.layer3 = nn.Conv2d(32, 64, 3, padding=1, stride=1)
        self.B3 = nn.BatchNorm2d(64)
        self.layer4 = nn.Conv2d(64, 64, 3, padding=1, stride=1)
        self.B4 = nn.BatchNorm2d(64)
        self.fc = nn.Linear(64 * (224 // 4) * (224 // 4), output_size)



    def forward(self, x):
        
        x = self.B1(F.leaky_relu(self.layer1(x)))
        x =  self.Maxpool(F.leaky_relu(self.layer2(x)))
        x=self.B2(x)
        x=self.B3(F.leaky_relu(self.layer3(x)))
        x = self.B4(self.Maxpool(F.leaky_relu(self.layer4(x))))

        return self.fc(x.view(x.size(0),-1))

#CNN with slight variations: Variant-1
# increased the kernel size to 5 and padding to 2
class Variant1_KernelSize(nn.Module):
    def __init__(self,input_size, hidden_size, output_size):
        super().__init__()
        self.layer1=nn.Conv2d(1,32,5,padding=2,stride=1)
        self.B1 = nn.BatchNorm2d(32)
        self.layer2 = nn.Conv2d(32, 32, 5, padding=2, stride=1)
        self.B2 = nn.BatchNorm2d(32)
        self.Maxpool=nn.MaxPool2d(2)
        self.layer3 = nn.Conv2d(32, 64, 5, padding=2, stride=1)
        self.B3 = nn.BatchNorm2d(64)
        self.layer4 = nn.Conv2d(64, 64, 5, padding=2, stride=1)
        self.B4 = nn.BatchNorm2d(64)
        self.fc = nn.Linear(64 * (224 // 4) * (224 // 4), output_size)


    def forward(self, x):        
        x = self.B1(F.leaky_relu(self.layer1(x)))
        x =  self.Maxpool(F.leaky_relu(self.layer2(x)))
        x=self.B2(x)
        x=self.B3(F.leaky_relu(self.layer3(x)))
        x = self.B4(self.Maxpool(F.leaky_relu(self.layer4(x))))
        return self.fc(x.view(x.size(0),-1))

#CNN Variant-2
#Added another layer to the convolutional network
class Variant2_LayerVariation(nn.Module):
    def __init__(self,input_size, hidden_size, output_size):
        super().__init__()

        self.layer1=nn.Conv2d(1,32,3,padding=1,stride=1)
        self.B1 = nn.BatchNorm2d(32)
        self.layer2 = nn.Conv2d(32, 32, 3, padding=1, stride=1)
        self.B2 = nn.BatchNorm2d(32)
        self.Maxpool=nn.MaxPool2d(2)
        self.layer3 = nn.Conv2d(32, 64, 3, padding=1, stride=1)
        self.B3 = nn.BatchNorm2d(64)
        self.layer4 = nn.Conv2d(64, 64, 3, padding=1, stride=1)
        self.B4 = nn.BatchNorm2d(64)
        self.layer5 = nn.Conv2d(64,128,3,padding=1, stride=1)
        self.B5 = nn.BatchNorm2d(128)
        self.fc = nn.Linear(128 * (224 // 8) * (224 // 8), output_size)



    def forward(self, x):        
        x = self.B1(F.leaky_relu(self.layer1(x)))
        x =  self.Maxpool(F.leaky_relu(self.layer2(x)))
        x=self.B2(x)
        x=self.B3(F.leaky_relu(self.layer3(x)))
        x = self.B4(self.Maxpool(F.leaky_relu(self.layer4(x))))
        x=self.B5(self.Maxpool(F.leaky_relu(self.layer5(x))))
        return self.fc(x.view(x.size(0),-1))

#Model for the Main Architecture that calculates running loss, validation accuracy and test accurcay
# Saves the best model.      
def MultiLayerModel(input_size,hidden_size,output_size,num_epochs,train_loader, val_loader, test_loader,patience=5):
        #epochs = 50
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = MultiLayerFCNet(input_size, hidden_size, output_size)
    model = nn.DataParallel(model)
    model.to(device)

    best_val_loss = float('inf')
    consecutive_no_improvement = 0
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    BestACC=0
    for epoch in range(num_epochs):
        running_loss = 0
        for instances, labels in train_loader:
            optimizer.zero_grad()
            output = model(instances)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print("main architecture running loss: ",running_loss / len(train_loader))

        model.eval()
        with torch.no_grad():
            allsamps=0
            val_loss = 0
            rightPred=0
            for instances, labels in val_loader:
                output = model(instances)
                loss = criterion(output, labels)
                val_loss += loss.item()
                _, predicted_class = torch.max(output, 1)
                allsamps += output.size(0)
                rightPred += (predicted_class == labels).sum().item()
            val_loss /= len(val_loader)
            val_accuracy = rightPred / allsamps
            print(f'Validation Accuracy of Main Architecture: {val_accuracy * 100:.2f}%')
            #print("val loss : ",val_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                consecutive_no_improvement = 0
            else:
                consecutive_no_improvement += 1
            #print("consecutive_no_improvement : ",consecutive_no_improvement)
            if consecutive_no_improvement >= patience:
                print(f"Early stopping at epoch {epoch + 1} due to no improvement in validation loss.")
                break

            if val_accuracy > BestACC:
                BestACC = val_accuracy
                torch.save(model.state_dict(), 'model_main')

        model.eval()
        with torch.no_grad():
            all_samples = 0
            correct_predictions = 0

            for instances, labels in test_loader:
                output = model(instances)
                _, predicted_class = torch.max(output, 1)
                all_samples += output.size(0)
                correct_predictions += (predicted_class == labels).sum().item()

            test_accuracy = correct_predictions / all_samples
            print(f'Test Accuracy of Main Architecture: {test_accuracy * 100:.2f}%')

        print(f"Best Validation Accuracy of Main Architecture: {BestACC * 100:.2f}%")
                
        model.train()

#Model for the variant-1 that calculates running loss, validation accuracy and test accurcay
# Saves the best model based on accuracy.  
def Variant1Model(input_size,hidden_size,output_size,num_epochs,train_loader, val_loader, test_loader,patience=5):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Variant1_KernelSize(input_size, hidden_size, output_size)
    model = nn.DataParallel(model)
    model.to(device)

    best_val_loss = float('inf')
    consecutive_no_improvement = 0
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    BestACC=0
    for epoch in range(num_epochs):
        running_loss = 0
        for instances, labels in train_loader:
            optimizer.zero_grad()
            output = model(instances)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print("variant-1 running loss : ",running_loss / len(train_loader))

        model.eval()
        with torch.no_grad():
            allsamps=0
            val_loss = 0
            rightPred=0
            for instances, labels in val_loader:
                output = model(instances)
                loss = criterion(output, labels)
                val_loss += loss.item()
                _, predicted_class = torch.max(output, 1)
                allsamps += output.size(0)
                rightPred += (predicted_class == labels).sum().item()
            val_loss /= len(val_loader)
            val_accuracy = rightPred / allsamps
            print(f'Validation Accuracy of variant-1 Architecture: {val_accuracy * 100:.2f}%')

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                consecutive_no_improvement = 0
            else:
                consecutive_no_improvement += 1
            #print("consecutive_no_improvement : ",consecutive_no_improvement)
            if consecutive_no_improvement >= patience:
                print(f"Early stopping at epoch {epoch + 1} due to no improvement in validation loss.")
                break

            if val_accuracy > BestACC:
                BestACC = val_accuracy
                torch.save(model.state_dict(), 'model_variant1')

        model.eval()
        with torch.no_grad():
            all_samples = 0
            correct_predictions = 0

            for instances, labels in test_loader:
                output = model(instances)
                _, predicted_class = torch.max(output, 1)
                all_samples += output.size(0)
                correct_predictions += (predicted_class == labels).sum().item()

            test_accuracy = correct_predictions / all_samples
            print(f'Test Accuracy of variant-1 Architecture: {test_accuracy * 100:.2f}%')

        print(f"Best Validation Accuracy of variant-1 Architecture: {BestACC * 100:.2f}%")
                
        model.train()

#Model for the variant-2 that calculates running loss, validation accuracy and test accurcay
# Saves the best model based on accuracy.  
def variant2Model(input_size,hidden_size,output_size,num_epochs,train_loader, val_loader, test_loader,patience=5):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Variant1_KernelSize(input_size, hidden_size, output_size)
    model = nn.DataParallel(model)
    model.to(device)

    best_val_loss = float('inf')
    consecutive_no_improvement = 0
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    BestACC=0
    for epoch in range(num_epochs):
        running_loss = 0
        for instances, labels in train_loader:
            optimizer.zero_grad()
            output = model(instances)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print("variant-2 running loss : ",running_loss / len(train_loader))

        model.eval()
        with torch.no_grad():
            allsamps=0
            val_loss = 0
            rightPred=0
            for instances, labels in val_loader:
                output = model(instances)
                loss = criterion(output, labels)
                val_loss += loss.item()
                _, predicted_class = torch.max(output, 1)
                allsamps += output.size(0)
                rightPred += (predicted_class == labels).sum().item()
            val_loss /= len(val_loader)
            val_accuracy = rightPred / allsamps
            print(f'Validation Accuracy of variant-2 Architecture: {val_accuracy * 100:.2f}%')

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                consecutive_no_improvement = 0
            else:
                consecutive_no_improvement += 1
            #print("consecutive_no_improvement : ",consecutive_no_improvement)
            if consecutive_no_improvement >= patience:
                print(f"Early stopping at epoch {epoch + 1} due to no improvement in validation loss.")
                break

            if val_accuracy > BestACC:
                BestACC = val_accuracy
                torch.save(model.state_dict(), 'model_variant2')

        model.eval()
        with torch.no_grad():
            all_samples = 0
            correct_predictions = 0

            for instances, labels in test_loader:
                output = model(instances)
                _, predicted_class = torch.max(output, 1)
                all_samples += output.size(0)
                correct_predictions += (predicted_class == labels).sum().item()

            test_accuracy = correct_predictions / all_samples
            print(f'Test Accuracy of variant-2 Architecture: {test_accuracy * 100:.2f}%')

        print(f"Best Validation Accuracy of variant-2 Architecture: {BestACC * 100:.2f}%")
                
        model.train()




if __name__ == '__main__':

    batch_size = 64
    test_batch_size = 64
    input_size = 1 * 224 * 224  # 1 channel, 224*224 image size
    hidden_size = 50  
    output_size = 4 
    num_epochs = 25

    train_loader, val_loader, test_loader = dataset_loader(batch_size)
    MultiLayerModel(input_size,hidden_size,output_size,num_epochs,train_loader, val_loader, test_loader)
    #Variant1Model(input_size,hidden_size,output_size,num_epochs,train_loader, val_loader, test_loader)   
    #variant2Model(input_size,hidden_size,output_size,num_epochs,train_loader, val_loader, test_loader)