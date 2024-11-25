#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 22:44:17 2024

@author: gelenag
"""

import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_model_with_early_stopping(model, trainloader, testloader, batch_size, model_name, num_epochs=200, patience=5):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    #optimizer = optim.Adam(model.parameters(), lr=3e-4, weight_decay=5e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=60)
    # scheduler = CosineAnnealingWarmupRestarts(optimizer, first_cycle_steps=20, cycle_mult=1.0, max_lr=3e-4, min_lr=1e-6, warmup_steps=5, gamma=0.5)
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [60, 120, 160], gamma=0.2) 

    best_accuracy = 0.0
    epochs_without_improvement = 0

    train_loss_history = []
    test_loss_history = []
    train_accuracy_history = []
    test_accuracy_history = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_value_(model.parameters(), 0.1)
            optimizer.step()
            running_loss += loss.item() / batch_size

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss:.4f}')


        scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch [{epoch + 1}/{num_epochs}], Learning Rate: {current_lr:.6f}')

        test_accuracy, test_loss, train_accuracy = test_model(model)

        print(f'Epoch [{epoch + 1}/{num_epochs}], Test Loss: {test_loss:.4f}')

        test_loss_history.append(test_loss)
        train_loss_history.append(running_loss)
        test_accuracy_history.append(test_accuracy)
        train_accuracy_history.append(train_accuracy)

        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            epochs_without_improvement = 0

            # En iyi modelin checkpointini kaydet
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'best_accuracy': best_accuracy,
                'train_loss_history': train_loss_history,
                'test_loss_history': test_loss_history,
                'train_accuracy_history': train_accuracy_history,
                'test_accuracy_history': test_accuracy_history
            }
            torch.save(checkpoint, f'../checkpoints/{model_name}_best_model_checkpoint.pth')
            print(f'Best model saved at epoch {epoch + 1} with accuracy: {best_accuracy:.4f}')
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f'Early stopping triggered after {epoch + 1} epochs.')
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'best_accuracy': best_accuracy,
                'train_loss_history': train_loss_history,
                'test_loss_history': test_loss_history,
                'train_accuracy_history': train_accuracy_history,
                'test_accuracy_history': test_accuracy_history
            }
            torch.save(checkpoint, f'../checkpoints/{model_name}_last_model_checkpoint.pth')
            break

    return model

def test_model(model, trainloader, testloader, batch_size):
    model.eval()
    correct = 0
    total = 0
    running_test_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    correct_train = 0
    total_train = 0

    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            running_test_loss += loss.item() / batch_size

        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

    test_loss = running_test_loss
    accuracy = 100 * correct / total
    train_accuracy = 100 * correct_train / total_train

    print(f'Accuracy on the test images: {accuracy:.2f}%')
    print(f'Accuracy on the train images: {train_accuracy:.2f}%')

    return accuracy, test_loss, train_accuracy