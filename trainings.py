import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt


def train(net, trainloader, valloader, criterion, optimizer, device, epochs=20, save_path=None):

    train_metrics = {"running_loss": [],
                        "top1_accuracy": [],
                        "avg_loss": [],
                        "val_running_loss": [],
                        "val_top1_accuracy": [],
                        "val_avg_loss": [],
                        "best_val_loss": float('inf'),
                        "best_val_epoch": 0}
    

    best_val_loss = float('inf')  # Start with an infinitely large validation loss

    for epoch in range(epochs):  
        correct_top1 = 0
        running_loss = 0.0  # Reset per epoca
        correct_top1_val = 0
        running_loss_val = 0.0

        net.train()
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs, 1)
            correct_top1 += (predicted == labels).sum().item()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            

        net.eval()
        for inputs, labels in valloader:
        
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs, 1)
            correct_top1_val += (predicted == labels).sum().item()
            loss = criterion(outputs, labels)
            running_loss_val += loss.item()

        if running_loss_val < best_val_loss:
            best_val_loss = running_loss_val
            # Save the best model
            if save_path is not None:
                torch.save(net.state_dict(), os.path.join(save_path, f"state_dict.pth"))
                print(f"Best model saved at epoch {epoch}")
                train_metrics["best_val_loss"] = best_val_loss
                train_metrics["best_val_epoch"] = epoch 
            # torch.save(net.state_dict(), os.path.join(save_path, 'state_dict.pth'))
            # logger.info(f"Model weights saved to {save_path}/state_dict.pth")

        train_metrics["val_top1_accuracy"].append(100 * correct_top1_val / len(valloader.dataset))
        train_metrics["val_running_loss"].append(running_loss_val / len(valloader))

        train_metrics["top1_accuracy"].append(100 * correct_top1 / len(trainloader.dataset)) 
        train_metrics["running_loss"].append(running_loss / len(trainloader))
        

        print(f'Epoch {epoch + 1}, Avg Loss: {running_loss / len(trainloader)}, Top-1 Accuracy: {100 * correct_top1 / len(trainloader.dataset)}')
        print(f'Validation Avg Loss: {running_loss_val / len(valloader)}, Validation Top-1 Accuracy: {100 * correct_top1_val / len(valloader.dataset)}')

    
        if save_path is not None and (epoch%10==0 or epoch==epochs-1):

            _, ax = plt.subplots(2, 1, figsize=(10, 10))
            ax[0].plot(train_metrics["running_loss"])
            ax[0].plot(train_metrics["val_running_loss"])
            ax[0].legend(["Training Loss", "Validation Loss"])
            ax[0].set_title("Loss")
            ax[0].set_xlabel("Epoch")
            ax[0].set_ylabel("Loss")
            
            ax[1].plot(train_metrics["top1_accuracy"])
            ax[1].plot(train_metrics["val_top1_accuracy"])
            ax[1].legend(["Training Top-1 Accuracy", "Validation Top-1 Accuracy"])
            ax[1].set_title("Top-1 Accuracy")
            ax[1].set_xlabel("Epoch")
            ax[1].set_ylabel("Accuracy")
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_path, "training_metrics.png"))
            plt.close()

    return train_metrics


def train_dist(student, teacher, trainloader, valloader, criterion, optimizer, device, epochs=20, save_path=None, temperature=3, alpha=0.5):

    train_metrics = {"running_loss": [],
                        "top1_accuracy": [],
                        "avg_loss": [],
                        "val_running_loss": [],
                        "val_top1_accuracy": [],
                        "val_avg_loss": [],
                        "best_val_loss": float('inf'),
                        "best_val_epoch": 0}
    
    best_val_loss = float('inf')  # Start with an infinitely large validation loss
    
    # Ensure teacher model is in eval mode
    teacher.eval()

    for epoch in range(epochs):
        correct_top1 = 0
        running_loss = 0.0
        correct_top1_val = 0
        running_loss_val = 0.0

        student.train()

        # Training loop
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass for student and teacher
            student_outputs = student(inputs)
            with torch.no_grad():
                teacher_outputs = teacher(inputs)  # No gradient for teacher

            # Compute the hard-label loss (CrossEntropy) and soft-label loss (KL Divergence)
            hard_loss = criterion(student_outputs, labels)
            soft_loss = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(student_outputs / temperature, dim=1),
                                                            F.softmax(teacher_outputs / temperature, dim=1)) * (temperature ** 2)

            # Total loss is a weighted sum of hard loss and soft loss
            loss = alpha * hard_loss + (1 - alpha) * soft_loss

            # Backpropagation
            loss.backward()
            optimizer.step()

            # Update metrics
            _, predicted = torch.max(student_outputs, 1)
            correct_top1 += (predicted == labels).sum().item()
            running_loss += loss.item()

        # Validation loop
        student.eval()
        with torch.no_grad():
            for inputs, labels in valloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = student(inputs)
                _, predicted = torch.max(outputs, 1)
                correct_top1_val += (predicted == labels).sum().item()
                loss = criterion(outputs, labels)
                running_loss_val += loss.item()

        if running_loss_val < best_val_loss:
            best_val_loss = running_loss_val
            if save_path is not None:
                torch.save(student.state_dict(), os.path.join(save_path, f"state_dict.pth"))
                print(f"Best model saved at epoch {epoch}")
                train_metrics["best_val_loss"] = best_val_loss
                train_metrics["best_val_epoch"] = epoch 

        # Compute validation metrics
        train_metrics["val_top1_accuracy"].append(100 * correct_top1_val / len(valloader.dataset))
        train_metrics["val_running_loss"].append(running_loss_val / len(valloader))

        # Compute training metrics
        train_metrics["top1_accuracy"].append(100 * correct_top1 / len(trainloader.dataset)) 
        train_metrics["running_loss"].append(running_loss / len(trainloader))

        # Print training progress
        print(f'Epoch {epoch + 1}, Avg Loss: {running_loss / len(trainloader)}, Top-1 Accuracy: {100 * correct_top1 / len(trainloader.dataset)}')
        print(f'Validation Avg Loss: {running_loss_val / len(valloader)}, Validation Top-1 Accuracy: {100 * correct_top1_val / len(valloader.dataset)}')

        # Save plots every 10 epochs or at the last epoch
        if save_path is not None and (epoch % 10 == 0 or epoch == epochs - 1):
            _, ax = plt.subplots(2, 1, figsize=(10, 10))
            ax[0].plot(train_metrics["running_loss"])
            ax[0].plot(train_metrics["val_running_loss"])
            ax[0].legend(["Training Loss", "Validation Loss"])
            ax[0].set_title("Loss")
            ax[0].set_xlabel("Epoch")
            ax[0].set_ylabel("Loss")

            ax[1].plot(train_metrics["top1_accuracy"])
            ax[1].plot(train_metrics["val_top1_accuracy"])
            ax[1].legend(["Training Top-1 Accuracy", "Validation Top-1 Accuracy"])
            ax[1].set_title("Top-1 Accuracy")
            ax[1].set_xlabel("Epoch")
            ax[1].set_ylabel("Accuracy")

            plt.tight_layout()
            plt.savefig(os.path.join(save_path, "training_metrics.png"))
            plt.close()

    return train_metrics


def test(net, testloader, criterion, device):

    correct_top1 = 0
    correct_top5 = 0
    total = 0
    test_loss = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)

            # Calcolo della perdita per il batch
            test_loss_batch = criterion(outputs, labels)
            test_loss += test_loss_batch.item()

            # Calcolo Top-1 (massima probabilità)
            _, predicted = torch.max(outputs, 1)
            correct_top1 += (predicted == labels).sum().item()

            # Calcolo Top-5
            _, top5_pred = outputs.topk(5, dim=1, largest=True, sorted=True)
            correct_top5 += (top5_pred == labels.view(-1, 1)).sum().item()

            total += labels.size(0)

    # Calcolo delle accuratezze finali
    top1_accuracy = 100 * correct_top1 / total
    top5_accuracy = 100 * correct_top5 / total
    avg_loss = test_loss / len(testloader)

    print(f'Accuracy of the network on the {total} test images from {len(testloader)} batches: \n'+
          f'Top-1 = {top1_accuracy}%, Top-5 = {top5_accuracy}%, Loss: {avg_loss}')
 
    return {"top1_accuracy": top1_accuracy, "top5_accuracy": top5_accuracy, "avg_loss": avg_loss}

