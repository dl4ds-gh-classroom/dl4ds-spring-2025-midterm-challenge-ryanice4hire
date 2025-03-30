import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os
import numpy as np
import pandas as pd
from tqdm.auto import tqdm  # For progress bars
import wandb
import json

################################################################################
# Define a one epoch training function
################################################################################
def train(epoch, model, trainloader, optimizer, criterion, scheduler, CONFIG):
    """Train one epoch."""
    device = CONFIG["device"]
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(trainloader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']} [Train]", leave=False)
    for i, (inputs, labels) in enumerate(progress_bar):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()  # Step One Cycle scheduler per batch

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        progress_bar.set_postfix({"loss": running_loss/(i+1), "acc": 100.*correct/total})
    train_loss = running_loss/len(trainloader)
    train_acc = 100.*correct/total
    return train_loss, train_acc

################################################################################
# Define a validation function
################################################################################
def validate(model, valloader, criterion, device):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        progress_bar = tqdm(valloader, desc="[Validate]", leave=False)
        for i, (inputs, labels) in enumerate(progress_bar):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            progress_bar.set_postfix({"loss": running_loss/(i+1), "acc": 100.*correct/total})
    val_loss = running_loss/len(valloader)
    val_acc = 100.*correct/total
    return val_loss, val_acc

def main():
    CONFIG = {
        "model": "ResNet50",
        "batch_size": 64,
        "learning_rate": 0.0001,
        "epochs": 35,
        "num_workers": 12,  # Use 12 workers for DataLoader
        # Use GPU if available, otherwise fallback to CPU
        "device": "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu",
        "data_dir": "./data",
        "ood_dir": "./data/ood-test",
        "wandb_project": "sp25-ds542-challenge",
        #"seed": 42,
    }
    import pprint
    print("\nCONFIG Dictionary:")
    pprint.pprint(CONFIG)

    # Force PyTorch to use 12 CPU threads for parallelism.
    torch.set_num_threads(12)
    torch.set_num_interop_threads(12)

    ############################################################################
    # Data Transformation
    # For ResNet50, we typically need larger images (e.g. 224x224) and ImageNet normalization.
    # Also, we fix the transform order so that RandomErasing is applied after ToTensor.
    ############################################################################
    transform_train = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.RandomErasing(),  # Now applied on tensor
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    ############################################################################
    # Data Loading
    ############################################################################
    full_trainset = torchvision.datasets.CIFAR100(root=CONFIG["data_dir"], train=True,
                                                  download=True, transform=transform_train)
    train_size = int(0.9 * len(full_trainset))
    val_size = len(full_trainset) - train_size
    trainset, valset = torch.utils.data.random_split(full_trainset, [train_size, val_size])
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=CONFIG["batch_size"],
                                              shuffle=True, num_workers=CONFIG["num_workers"])
    valloader = torch.utils.data.DataLoader(valset, batch_size=CONFIG["batch_size"],
                                            shuffle=False, num_workers=CONFIG["num_workers"])
    testset = torchvision.datasets.CIFAR100(root=CONFIG["data_dir"], train=False,
                                            download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=CONFIG["batch_size"],
                                             shuffle=False, num_workers=CONFIG["num_workers"])
    
    ############################################################################
    # Model Instantiation: Use pretrained ResNet50 and modify its final layer.
    ############################################################################
    model = torchvision.models.resnet50(weights="DEFAULT") ## Instantialize the ResNet50 Model
   
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 100)
    for x in model.parameters():
        x.requires_grad = True
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, 100)
    )
    model = model.to(CONFIG["device"])
    print("\nModel summary:")
    print(model)

    ############################################################################
    # Loss, Optimizer, and One Cycle LR Scheduler
    ############################################################################
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"])
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, 
                                              max_lr=CONFIG["learning_rate"],
                                              steps_per_epoch=len(trainloader),
                                              epochs=CONFIG["epochs"])
    
    wandb.init(project=CONFIG["wandb_project"], config=CONFIG)
    wandb.watch(model)

    ############################################################################
    # Training Loop
    ############################################################################
    best_val_acc = 0.0
    for epoch in range(CONFIG["epochs"]):
        train_loss, train_acc = train(epoch, model, trainloader, optimizer, criterion, scheduler, CONFIG)
        val_loss, val_acc = validate(model, valloader, criterion, CONFIG["device"])
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": optimizer.param_groups[0]["lr"]
        })
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")
            wandb.save("best_model.pth")
    
    wandb.finish()

    ############################################################################
    # Evaluation
    ############################################################################
    import eval_cifar100
    import eval_ood
    predictions, clean_accuracy = eval_cifar100.evaluate_cifar100_test(model, testloader, CONFIG["device"])
    print(f"Clean CIFAR-100 Test Accuracy: {clean_accuracy:.2f}%")
    all_predictions = eval_ood.evaluate_ood_test(model, CONFIG)
    submission_df_ood = eval_ood.create_ood_df(all_predictions)
    submission_df_ood.to_csv("submission_ood50.csv", index=False)
    print("submission_ood50.csv created successfully.")

if __name__ == '__main__':
    main()
