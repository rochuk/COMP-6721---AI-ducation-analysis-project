import torch
import numpy as np
import random

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

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
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report

def dataset_loader(batch_size):
    normalize = transforms.Normalize(mean=[0.5], std=[0.5])
    # Loading training dataset with data augmentation techniques
    dataset = datasets.ImageFolder(root='/Users/roshinichukkapalli/Desktop/Senior',
                                   transform=transforms.Compose([
                                       transforms.Resize((224, 224)),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.RandomCrop(224, 4),
                                       transforms.Grayscale(num_output_channels=1),
                                       transforms.ToTensor(),
                                       normalize
                                   ]))
    train_dataset, test_dataset = train_test_split(dataset, test_size=0.3, random_state=42)
    train_loader = td.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_loader = td.DataLoader(test_dataset, batch_size=batch_size, shuffle=True,pin_memory=True)
    return train_loader, test_loader

class MultiLayerFCNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.layer1 = nn.Conv2d(1, 32, 3, padding=1, stride=1)
        self.B1 = nn.BatchNorm2d(32)
        self.layer2 = nn.Conv2d(32, 32, 3, padding=1, stride=1)
        self.B2 = nn.BatchNorm2d(32)
        self.Maxpool = nn.MaxPool2d(2)
        self.layer3 = nn.Conv2d(32, 64, 3, padding=1, stride=1)
        self.B3 = nn.BatchNorm2d(64)
        self.layer4 = nn.Conv2d(64, 64, 3, padding=1, stride=1)
        self.B4 = nn.BatchNorm2d(64)
        self.fc = nn.Linear(64 * (224 // 4) * (224 // 4), output_size)

    def forward(self, x):
        x = self.B1(F.leaky_relu(self.layer1(x)))
        x = self.Maxpool(F.leaky_relu(self.layer2(x)))
        x = self.B2(x)
        x = self.B3(F.leaky_relu(self.layer3(x)))
        x = self.B4(self.Maxpool(F.leaky_relu(self.layer4(x))))
        return self.fc(x.view(x.size(0), -1))
def MultiLayerModel(input_size, hidden_size, output_size, num_epochs, train_loader, val_loader, patience=5):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = MultiLayerFCNet(input_size, hidden_size, output_size)
    model = nn.DataParallel(model)
    model.to(device)

    best_val_loss = float('inf')
    consecutive_no_improvement = 0
    criteria = nn.CrossEntropyLoss()
    a_optimizer = optim.Adam(model.parameters(), lr=0.01)
    best_accuracy = 0

    # Initialize lists for storing true and predicted labels
    all_true_labels = []
    all_pred_labels = []

    for epoch in range(num_epochs):
        running_loss = 0
        for instances, labels in train_loader:
            a_optimizer.zero_grad()
            output = model(instances)
            loss = criteria(output, labels)
            loss.backward()
            a_optimizer.step()
            running_loss += loss.item()
        print("Main architecture running loss: ", running_loss / len(train_loader))

        model.eval()
        with torch.no_grad():
            all_samps = 0
            val_loss = 0
            correct_pred = 0
            for instances, labels in val_loader:
                output = model(instances)
                loss = criteria(output, labels)
                val_loss += loss.item()
                _, predicted_class = torch.max(output, 1)
                all_samps += output.size(0)
                correct_pred += (predicted_class == labels).sum().item()

                # Store true and predicted labels
                all_true_labels.extend(labels.cpu().numpy())
                all_pred_labels.extend(predicted_class.cpu().numpy())

            val_loss /= len(val_loader)
            val_accuracy = correct_pred / all_samps
            print(f'Validation Accuracy of Main Architecture: {val_accuracy * 100:.2f}%')
            print("Validation Loss: ", val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                consecutive_no_improvement = 0
            else:
                consecutive_no_improvement += 1

            if consecutive_no_improvement >= patience:
                print(f"Early stopping at epoch {epoch + 1} due to no improvement in validation loss.")
                break

            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                torch.save(model.state_dict(), 'model_kfold')

    # After training, calculate and print evaluation metrics on the validation set
    print("\nValidation Metrics:")
    print_confusion_matrix(all_true_labels, all_pred_labels)
    print_evaluation_metrics(all_true_labels, all_pred_labels)

    

# Function to print confusion matrix
def print_confusion_matrix(true_labels, pred_labels):
    class_names = ['Anger     :', 'Neutral   :', 'Engaged   :', 'Bored     :']
    cm = confusion_matrix(true_labels, pred_labels)
    print("\nConfusion Matrix:")
    #print(cm)
    for i in range(len(class_names)):
            print(f"{class_names[i]}{cm[i]}")

# Function to print evaluation metrics
def print_evaluation_metrics(true_labels, pred_labels):
    accuracy = accuracy_score(true_labels, pred_labels)
    precision_macro = precision_score(true_labels, pred_labels, average='macro')
    recall_macro = recall_score(true_labels, pred_labels, average='macro')
    f1_macro = f1_score(true_labels, pred_labels, average='macro')
    precision_micro = precision_score(true_labels, pred_labels, average='micro')
    recall_micro = recall_score(true_labels, pred_labels, average='micro')
    f1_micro = f1_score(true_labels, pred_labels, average='micro')

    print("\nMetrics:")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("\nMacro-averaged Metrics:")
    print(f"Precision: {precision_macro:.4f}")
    print(f"Recall: {recall_macro:.4f}")
    print(f"F1-score: {f1_macro:.4f}")
    print("\nMicro-averaged Metrics:")
    print(f"Precision: {precision_micro:.4f}")
    print(f"Recall: {recall_micro:.4f}")
    print(f"F1-score: {f1_micro:.4f}")


def test_best_model(test_loader):
    #class_names = ['Anger     :', 'Neutral   :', 'Engaged   :', 'Bored     :']
    
    all_true_labels = []
    all_pred_labels = []
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = MultiLayerFCNet(input_size, hidden_size, output_size)
    model = nn.DataParallel(model)
    model.to(device)
    model.load_state_dict(torch.load('model_kfold'))
    model.eval()
    with torch.no_grad():
        all_samples = 0
        correct_predictions = 0
        for instances, labels in test_loader:
            output = model(instances)
            _, predicted_class = torch.max(output, 1)
            all_samples += output.size(0)
            correct_predictions += (predicted_class == labels).sum().item()
            
            all_true_labels.extend(labels.cpu().numpy())
            all_pred_labels.extend(predicted_class.cpu().numpy())
    test_accuracy = correct_predictions / all_samples
    print(f'Test Accuracy of Main Architecture: {test_accuracy * 100:.2f}%')
    print_confusion_matrix(all_true_labels, all_pred_labels)
    print_evaluation_metrics(all_true_labels, all_pred_labels)

if __name__ == '__main__':

    batch_size = 64
    test_batch_size = 64
    input_size = 1 * 224 * 224  # 1 channel, 224*224 image size
    hidden_size = 50  
    output_size = 4 
    num_epochs = 25

    train_loader, test_loader = dataset_loader(batch_size)
    #MultiLayerModel(input_size,hidden_size,output_size,num_epochs,train_loader, val_loader
    test_best_model(test_loader)
