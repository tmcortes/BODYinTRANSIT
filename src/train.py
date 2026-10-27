import copy
import json
import pickle 
import os
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import balanced_accuracy_score, f1_score

from dataset import DatasetManager
from models import RoSA


import torch.nn.functional as F

def compute_loss_weights(device, dm):

    # 1. Get all labels from the training set segments
    train_labels = [walk.label for walk in dm.trWalks]

    # 2. Count occurrences of each class
    class_counts = np.bincount(train_labels, minlength=3)

    # 3. Calculate weights (inverse frequency)
    class_weights = len(train_labels) / (len(class_counts) * (class_counts + 1e-6)) # Add epsilon
    
    # 4. Convert to a PyTorch tensor and move to the correct device
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
    print(f"Using class weights for the loss function: {class_weights_tensor.cpu().numpy().round(2)}")

    return class_weights_tensor

def train(id_, seed, args):

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset path
    folderPath = '../dataset/unified_db' 

    # CSV columns from the 17 accelerometers in Rokoko
    with open('accelerometer_columns.txt', 'r') as f:
        featuresIdx = [int(line.strip()) for line in f]

    num_sensors = int(len(featuresIdx)/3) if featuresIdx else 17

    splitPcts= {'train': 60, 'val': 20, 'test': 20}

    dm = DatasetManager(folderPath, id_, winLength=150, featuresIdx=featuresIdx, 
        splitPcts=splitPcts, label=args.task, seed=seed)
    
    train_dataset = dm.createDataset(mode='train')
    val_dataset = dm.createDataset(mode='val')
    test_dataset = dm.createDataset(mode='test')

    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    # Initialize model
    model = RoSA(freeze_harnet=True, num_classes=3).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(weight=compute_loss_weights(device, dm))
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # Training loop state
    train_losses, train_accuracies = [], []
    val_losses, val_accuracies, val_balanced_accuracies = [], [], []
    best_val_loss = float('inf')
    best_model_state = None
    best_epoch = -1
    patience = 10
    epochs_no_improve = 0

    for epoch in range(args.epochs):
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")

        for batch in pbar:
            X, y = batch
            X, y = X.to(device), y.to(device)

            optimizer.zero_grad()
            outputs, _ = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += y.size(0)
            train_correct += (predicted == y).sum().item()
            pbar.set_postfix({'loss': f"{train_loss / (pbar.n + 1):.4f}", 'acc': f"{train_correct / train_total:.4f}"})

        train_loss /= len(train_loader)
        train_accuracy = train_correct / train_total
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        # Validation
        model.eval()
        val_loss = 0
        all_val_preds, all_val_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                X, y = batch
                X, y = X.to(device), y.to(device)
                outputs, _ = model(X)
                loss = criterion(outputs, y)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                all_val_preds.extend(predicted.cpu().numpy())
                all_val_labels.extend(y.cpu().numpy())

        val_loss /= len(val_loader)
        val_accuracy = (np.array(all_val_preds) == np.array(all_val_labels)).mean()
        val_balanced_acc = balanced_accuracy_score(all_val_labels, all_val_preds)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        val_balanced_accuracies.append(val_balanced_acc)

        print(f"Epoch [{epoch+1}/{args.epochs}], "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Bal Acc: {val_balanced_acc:.4f}")


        # Perfect validation (some participants have the same label always)
        if val_balanced_acc >= 0.999:
            print(f"Validation Balanced Accuracy is 100%. Perfection achieved. Stopping training at epoch {epoch + 1}.")
            
            best_epoch = epoch + 1
            best_model_state = {
                'epoch': best_epoch,
                'model_state_dict': copy.deepcopy(model.state_dict()),
                'optimizer_state_dict': copy.deepcopy(optimizer.state_dict()),
                'val_loss': val_loss,
                'val_accuracy': val_accuracy,
                'val_balanced_accuracy': val_balanced_acc
            }
            break

        # check for improving val loss and keep track of it for early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            best_model_state = {
                'epoch': best_epoch,
                'model_state_dict': copy.deepcopy(model.state_dict()),
                'optimizer_state_dict': copy.deepcopy(optimizer.state_dict()),
                'val_loss': best_val_loss,
                'val_accuracy': val_accuracy,
                'val_balanced_accuracy': val_balanced_acc
            }
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs.")
            break
    
    # ======= AFTER TRAINING: evaluate best val model on test set =======
    print("\n--- Evaluating best model on test set... ---")
    model.load_state_dict(best_model_state['model_state_dict'])
    model.eval()
    all_test_preds, all_test_labels = [], []
    running_attn_weights = torch.zeros(num_sensors, device=device)
    total_test_samples = 0

    with torch.no_grad():
        for batch in test_loader:
            X, y = batch
            X, y = X.to(device), y.to(device)
            outputs, attn_weights = model(X)

            running_attn_weights += attn_weights.sum(dim=0)
            total_test_samples += X.size(0)

            _, predicted = outputs.max(1)
            all_test_preds.extend(predicted.cpu().numpy())
            all_test_labels.extend(y.cpu().numpy())

    # calculate the average sensor importance
    mean_sensor_importance = (running_attn_weights / total_test_samples).cpu().numpy()
    
    # Calculate all final test metrics
    test_accuracy = (np.array(all_test_preds) == np.array(all_test_labels)).mean()
    test_balanced_acc = balanced_accuracy_score(all_test_labels, all_test_preds)
    test_macro_f1 = f1_score(all_test_labels, all_test_preds, average='macro', zero_division=0)
    

    print(f"Best model from Epoch {best_epoch} with Val Loss: {best_val_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Balanced Accuracy: {test_balanced_acc:.4f}")
    print(f"Test Macro F1-Score: {test_macro_f1:.4f}")
    print("----------------------------------------------------")

    # best_model_state['test_accuracy'] = test_accuracy
    # best_model_state['test_balanced_accuracy'] = test_balanced_acc
    # best_model_state['test_macro_f1'] = test_macro_f1
    # torch.save(best_model_state, f'checkpoints/best_model_{id_}_{seed}.pth') # Optional: save best model

    # # Plotting
    # plt.figure(figsize=(12, 5))
    # plt.subplot(1, 2, 1)
    # plt.plot(train_losses, label='Train Loss')
    # plt.plot(val_losses, label='Validation Loss')
    # plt.axvline(x=best_epoch-1, color='r', linestyle='--', label=f'Best Epoch ({best_epoch})')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.legend()

    # plt.subplot(1, 2, 2)
    # plt.plot(train_accuracies, label='Train Accuracy')
    # plt.plot(val_accuracies, label='Validation Accuracy')
    # plt.plot(val_balanced_accuracies, label='Validation Balanced Acc', linestyle='--')
    # plt.axhline(y=test_balanced_acc, color='g', linestyle='-.', label=f'Final Test Bal. Acc ({test_balanced_acc:.2f})')
    # plt.axvline(x=best_epoch-1, color='r', linestyle='--', label=f'Best Epoch ({best_epoch})')
    # plt.xlabel('Epoch')
    # plt.ylabel('Accuracy')
    # plt.legend()
    # plt.title('Training and Validation Metrics')

    # plt.tight_layout()
    # plt.savefig(f"../results/training_plot_{id_}_seed{seed}.png")
    # plt.close()

    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_accuracies": train_accuracies,
        "val_accuracies": val_accuracies,
        "val_balanced_accuracies": val_balanced_accuracies,
        "test_predictions": all_test_preds,
        "test_labels": all_test_labels,
        "test_accuracy": test_accuracy,
        "test_balanced_accuracy": test_balanced_acc,
        "test_macro_f1": test_macro_f1,
        "best_epoch": best_epoch,
        "sensor_importance": mean_sensor_importance
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the RoSA model")
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    parser.add_argument('--task', choices=['condition', 'weight', 'happiness'],
                        default='happiness', help='Prediction target')
    parser.add_argument('--output_dir', type=str, default='../results',
                        help='Directory for participant result files')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open("subjectIds.json", "r") as f:
        subjectIds = json.load(f)

    seeds = [1,2,3,4,5,6,7,8,9,10]
    for id_ in subjectIds:
        results = {}
        for seed in seeds:
            key = f"{id_}_seed{seed}"
            print(f"\n===== Training for {key} =====")
            result = train(id_, seed, args)
            results[key] = result

        with open(os.path.join(args.output_dir, f"results_{id_}.pkl"), "wb") as f:
            pickle.dump(results, f)