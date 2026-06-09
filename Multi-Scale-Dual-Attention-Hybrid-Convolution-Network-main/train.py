from tqdm import tqdm
import os
import random
import time
import datetime
import numpy as np
import albumentations as A
import cv2
from PIL import Image
from glob import glob
import torch
import warnings
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from lib.utils import seeding, create_dir, print_and_save, shuffling, epoch_time,calculate_metrics
from lib.metrics import *
from lib.loss import *
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import numpy as np
import argparse
from model.resunet import*


def load_data(dataset_name):
    random.seed(42)
    path = f"../Segmentation/Dataset/{dataset_name}"
    images_path = os.path.join(path, "Images")
    masks_path = os.path.join(path, "Masks")
    dirs = sorted(os.listdir(images_path))  # Use image directory for listing directories
    print("Number of directories:", len(dirs))
    random.shuffle(dirs)
    # Split data into training, validation, and testing sets
    train_dirs, temp_dirs = train_test_split(dirs, test_size=0.3, random_state=42)
    valid_dirs, test_dirs = train_test_split(temp_dirs, test_size=0.5, random_state=42)
    def load_dataset(dir_list):
        x, y = [], []
            for name in dir_list:
                image = Image.open(os.path.join(images_path, name))
                mask = Image.open(os.path.join(masks_path, name))
                x.append(np.array(image))
                y.append(np.array(mask))
            return x, y
        train_data = load_dataset(train_dirs)
        valid_data = load_dataset(valid_dirs)
        test_data = load_dataset(test_dirs)
        return [train_data, valid_data, test_data]


class DATASET(Dataset):
    def __init__(self, images_path, masks_path, size, transform=None):
        self.images_path = images_path
        self.masks_path = masks_path
        self.size = size
        self.transform = transform
        self.n_samples = len(images_path)

    def __getitem__(self, index):
        image = self.images_path[index]
        mask = self.masks_path[index]
        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]
        image = cv2.resize(image, self.size)
        image = np.expand_dims(image, axis=0)
        image = image / 255.0

        mask = cv2.resize(mask, self.size)
        mask = np.expand_dims(mask, axis=0)
        mask = mask / 255.0

        return image, mask

    def __len__(self):
        return self.n_samples


def train(model, loader, optimizer, loss_fn, device):
    model.train()
    epoch_loss = 0.0
    epoch_metrics = np.zeros(9)  # Assuming there are 6 metrics in your calculation
    for i, (x, y) in enumerate(loader):
        x = x.to(device, dtype=torch.float32)
        y = y.to(device, dtype=torch.float32)
        optimizer.zero_grad()
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        """ Calculate the metrics """
        y_pred = torch.sigmoid(y_pred)
        # Using NumPy to calculate metrics in batch
        batch_metrics = np.array([calculate_metrics(yt, yp) for yt, yp in zip(y, y_pred)])
        # Accumulate batch metrics
        epoch_metrics += np.mean(batch_metrics, axis=0)
    # Average metrics over all batches
    epoch_loss = epoch_loss/len(loader)
    epoch_metrics = epoch_metrics/len(loader)
    return epoch_loss, epoch_metrics


def evaluate(model, loader, loss_fn, device):
    model.eval()
    epoch_loss = 0.0
    epoch_metrics = np.zeros(9)  # Assuming there are 6 metrics in your calculation
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            epoch_loss += loss.item()
            """ Calculate the metrics """
            y_pred = torch.sigmoid(y_pred)
            # Using NumPy to calculate metrics in batch
            batch_metrics = np.array([calculate_metrics(yt, yp) for yt, yp in zip(y, y_pred)])
            # Accumulate batch metrics
            epoch_metrics += np.mean(batch_metrics, axis=0)
        # Average metrics over all batches
        epoch_loss = epoch_loss / len(loader)
        epoch_metrics = epoch_metrics / len(loader)

        return epoch_loss, epoch_metrics


def evaluate(model, loader, loss_fn, device):
    model.eval()
    epoch_loss = 0.0
    epoch_jac = 0.0
    epoch_dice = 0.0
    epoch_recall = 0.0
    epoch_precision = 0.0
    epoch_f2=0.0
    epoch_accuracy = 0.0

    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)

            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            epoch_loss += loss.item()

            """ Calculate the metrics """
            batch_jac = []
            batch_dice = []
            batch_recall = []
            batch_precision = []
            batch_f2 = []
            batch_accuracy = []

            y_pred = torch.sigmoid(y_pred)
            for yt, yp in zip(y, y_pred):
                score = calculate_metrics(yt, yp)
                batch_jac.append(score[0])
                batch_dice.append(score[1])
                batch_recall.append(score[2])
                batch_precision.append(score[3])
                batch_f2.append(score[4])
                batch_accuracy.append(score[5])

            epoch_jac += np.mean(batch_jac)
            epoch_dice += np.mean(batch_dice)
            epoch_recall += np.mean(batch_recall)
            epoch_precision += np.mean(batch_precision)
            epoch_f2 += np.mean(batch_f2)
            epoch_accuracy += np.mean(batch_accuracy)

        epoch_loss = epoch_loss / len(loader)
        epoch_dice = epoch_dice / len(loader)
        epoch_accuracy = epoch_accuracy / len(loader)
        return epoch_loss, [ epoch_dice, epoch_accuracy]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=5, type=int, metavar='N',
                        help='number of total epochs to run(default: 400)')
    parser.add_argument('-b', '--batch_size', default=4, type=int,
                        metavar='N', help='batch size (default: 1)')
    parser.add_argument('--learning_rate', default=1e-3, type=float,
                        metavar='lr', help='initial learning rate (default: 0.001)')
    parser.add_argument('--early_stopping_patience', default=15, type=float,
                        metavar='W', help='early_stopping_patience (default: 50)')
    parser.add_argument('--modelname', default='build_resunetplusplus', type=str,
                        help='type of model')
    parser.add_argument('--imgsize', type=int, default=256)
    parser.add_argument('--datasetname', default='mydataset', type=str,
                        help='type of dataset')

    args = parser.parse_args()
    batch_size= args.batch_size
    num_epochs = args.epochs
    lr = args.learning_rate
    early_stopping_patience= args.early_stopping_patience
    model_name = args.modelname
    image_size = args.imgsize
    dataset_name = args.datasetname

    """ Seeding """
    seeding(42)
    """ Directories """
    create_dir("files")


    train_log_path = f"files/{dataset_name}-{model_name}.txt"
    if os.path.exists(train_log_path):
        print("Log file exists")
    else:
        train_log = open(f"files/{dataset_name}-{model_name}.txt", "w")
        train_log.write("\n")
        train_log.close()

    """ Record Date & Time """
    datetime_object = str(datetime.datetime.now())
    print_and_save(train_log_path, datetime_object)
    print("")

    if model_name == "build_resunetplusplus":
        model= build_resunetplusplus(input_channels=1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    checkpoint_path = f"files/checkpoint-{dataset_name}-{model_name}.pth"

    data_str = f"Image Size: {(image_size, image_size)}\nBatch Size: {batch_size}\nLR: {lr}\nEpochs: {num_epochs}\n"
    data_str += f"Early Stopping Patience: {early_stopping_patience}\n"
    print_and_save(train_log_path, data_str)

    """ Dataset """
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_data(dataset_name)
    train_x, train_y = shuffling(train_x, train_y)

    data_str = f"Dataset Size:\nTrain: {len(train_x)} - Valid: {len(valid_x)} - Test: {len(test_x)}\n"
    print_and_save(train_log_path, data_str)

    """ data augmentation: Transforms """
    transform =  A.Compose([
        A.Rotate(limit=35, p=0.3),
        A.HorizontalFlip(p=0.3),
        A.VerticalFlip(p=0.3),
        A.CoarseDropout(p=0.3, max_holes=10, max_height=32, max_width=32)
    ])

    """ Dataset and loader """
    train_dataset = DATASET(train_x, train_y, (image_size, image_size), transform=transform)
    valid_dataset = DATASET(valid_x, valid_y, (image_size, image_size), transform=None)


    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,num_workers=8)

    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size,
        shuffle=False,num_workers=8)

    """ Model """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True)
    loss_fn =DiceBCELoss()
    loss_name = "DiceBCELoss"
    data_str = f"Optimizer: Adam\nLoss: {loss_name}\n"
    print_and_save(train_log_path, data_str)

    """ Training the model """
    start_time = time.time()
    best_valid_metrics = 0.0
    early_stopping_count = 0
    train_losses, train_accuracies = [], []
    valid_losses, valid_accuracies = [], []

    for epoch in range(num_epochs):
        start_time = time.time()
        train_loss, train_metrics = train(model, tqdm(train_loader), optimizer, loss_fn, device)
        valid_loss, valid_metrics = evaluate(model, tqdm(valid_loader), loss_fn, device)
        scheduler.step(valid_loss)

        if valid_metrics[1] > best_valid_metrics:
            data_str = f"Valid dice improved from {best_valid_metrics:2.4f} to {valid_metrics[1]:2.4f}. Saving checkpoint: {checkpoint_path}"
            print_and_save(train_log_path, data_str)

            best_valid_metrics = valid_metrics[1]
            torch.save(model.state_dict(), checkpoint_path)
            early_stopping_count = 0

        elif valid_metrics[1] < best_valid_metrics:
            early_stopping_count += 1

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        train_losses.append(train_loss)
        train_accuracies.append(train_metrics[0])
        valid_losses.append(valid_loss)
        valid_accuracies.append(valid_metrics[0])


        data_str = f"Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s\n"
        data_str += f"\tTrain Loss: {train_loss:.4f}  - Dice: {train_metrics[0]:.4f}- accuracy: {train_metrics[1]:.4f} \n"
        data_str += f"\t Val. Loss: {valid_loss:.4f}  - Dice: {train_metrics[0]:.4f}- accuracy: {valid_metrics[1]:.4f} \n"
        print_and_save(train_log_path, data_str)
        torch.save(model.state_dict(), checkpoint_path)

        if early_stopping_count == early_stopping_patience:
            data_str = f"Early stopping: validation loss stops improving from last {early_stopping_patience} continously.\n"
            print_and_save(train_log_path, data_str)
            break
