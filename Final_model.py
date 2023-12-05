import torch
import numpy as np
from torch import optim, nn
import torch.nn.functional as F
import torch.utils.data as td
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report

# Function to load dataset
def dataset_loader():
    normalize = transforms.Normalize(mean=[0.5], std=[0.5])
    dataset = datasets.ImageFolder(root=r'/home/AI/AIdatasets/TrainDataset', transform=transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(224, 4),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        normalize
    ]))
    return dataset

# Creating CNN: main architecture
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

# Model for the Main Architecture that calculates running loss, validation accuracy, and test accuracy
# Saves the best model
def MultiLayerModel(input_size, hidden_size, output_size, num_epochs, dataset, batch_size, patience=5):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = MultiLayerFCNet(input_size, hidden_size, output_size)
    model = nn.DataParallel(model)
    model.to(device)
    criteria = nn.CrossEntropyLoss()
    a_optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Initialize KFold
    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    # Initialize lists for storing metrics for each fold
    fold_accuracies = []
    fold_precisions_macro = []
    fold_recalls_macro = []
    fold_f1_scores_macro = []
    fold_precisions_micro = []
    fold_recalls_micro = []
    fold_f1_scores_micro = []

    # Initialize a variable to store the best confusion matrix
    best_confusion_matrix = None
    best_confusion_matrix_accuracy = 0

    for fold, (train_indices, val_indices) in enumerate(kf.split(dataset)):
        # Create data loaders for this fold
        train_dataset = torch.utils.data.Subset(dataset, train_indices)
        val_dataset = torch.utils.data.Subset(dataset, val_indices)
        train_loader = td.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
        val_loader = td.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

        best_val_loss = float('inf')
        consecutive_no_improvement = 0
        best_accuracy = 0

        # Initialize lists for storing true and predicted labels
        all_true_labels = []
        all_pred_labels = []

        for epoch in range(num_epochs):
            running_loss = 0
            for instances, labels in train_loader:
                instances = instances.to(device)
                labels = labels.to(device)
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
                    instances = instances.to(device)
                    labels = labels.to(device)
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
                    torch.save(model.state_dict(), 'model_main')

        # After training, calculate and print evaluation metrics on the validation set
        print(f"\nValidation Metrics for Fold {fold + 1}:")
        accuracy, precision_macro, recall_macro, f1_macro, precision_micro, recall_micro, f1_micro = print_evaluation_metrics(all_true_labels, all_pred_labels)

        
        # Calculate confusion matrix for this fold
        confusion_mat = confusion_matrix(all_true_labels, all_pred_labels)
        print(f"Confusion Matrix for Fold {fold + 1}:\n{confusion_mat}")

        # If this fold's accuracy is better than the best seen so far, update the best confusion matrix
        if accuracy > best_confusion_matrix_accuracy:
            best_confusion_matrix = confusion_mat
            best_confusion_matrix_accuracy = accuracy

        # Store metrics for this fold
        fold_accuracies.append(accuracy)
        fold_precisions_macro.append(precision_macro)
        fold_recalls_macro.append(recall_macro)
        fold_f1_scores_macro.append(f1_macro)
        fold_precisions_micro.append(precision_micro)
        fold_recalls_micro.append(recall_micro)
        fold_f1_scores_micro.append(f1_micro)

    # Print average metrics across all folds
    print("\nAverage Metrics Across All Folds:")
    print(f"Accuracy: {np.mean(fold_accuracies) * 100:.2f}%")
    print(f"Macro Precision: {np.mean(fold_precisions_macro):.4f}")
    print(f"Macro Recall: {np.mean(fold_recalls_macro):.4f}")
    print(f"Macro F1-score: {np.mean(fold_f1_scores_macro):.4f}")
    print(f"Micro Precision: {np.mean(fold_precisions_micro):.4f}")
    print(f"Micro Recall: {np.mean(fold_recalls_micro):.4f}")
    print(f"Micro F1-score: {np.mean(fold_f1_scores_micro):.4f}")

    # Print the best confusion matrix
    print(f"Best Confusion Matrix Across All Folds (Accuracy: {best_confusion_matrix_accuracy * 100:.2f}%):\n{best_confusion_matrix}")

        

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

    return accuracy, precision_macro, recall_macro, f1_macro, precision_micro, recall_micro, f1_micro

if __name__ == '__main__':
    batch_size = 64
    input_size = 1 * 224 * 224  # 1 channel, 224*224 image size
    hidden_size = 50
    output_size = 4
    num_epochs = 25
    dataset = dataset_loader()
    MultiLayerModel(input_size, hidden_size, output_size, num_epochs, dataset, batch_size)
