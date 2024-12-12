import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
# from torchvision import models
from models import model_dict, ensemble_of_models
import os
import matplotlib.pyplot as plt
from loaders import get_train_and_test_loader
import argparse
from trainings import train, train_dist, test
from attacks import pgd_attack, fgsm_attack
import torchvision
from torchvision import datasets



def test_with_adversarial(net, testloader, device, epsilon, criterion, attack_type = None, save_first=False):

    assert attack_type in ["fgsm", "pgd"], "Invalid attack type"

    print("\n ***Adversarial test***")

    correct_top1 = 0
    correct_top5 = 0
    total = 0
    saved_flag = False
    adv_loss = 0

    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        
        # Applica l'attacco FGSM
        if attack_type == "fgsm":
            adv_images = fgsm_attack(images, labels, net, epsilon, criterion)

        elif attack_type == "pgd":
            adv_images = pgd_attack(images, labels, net, epsilon, criterion)

        if save_first and not saved_flag:
            # Salva la prima immagine avversariale
            print("Saving the first adversarial image")
            combined_images = torch.cat((images, adv_images), dim=0)  # Combina immagini originali e avversariali
            grid = torchvision.utils.make_grid(combined_images, nrow=testloader.batch_size)
            torchvision.utils.save_image(grid.to('cpu'), "work/project/saved_fig/combined_image" + str(epsilon) + ".png")
            saved_flag = True
        
        # Ottieni le predizioni
        outputs = net(adv_images)
        adv_loss += criterion(outputs, labels).item()

        # Calcolo Top-1
        _, predicted = torch.max(outputs, 1)
        correct_top1 += (predicted == labels).sum().item()

        # Calcolo Top-5
        _, top5_pred = outputs.topk(5, dim=1, largest=True, sorted=True)
        correct_top5 += (top5_pred == labels.view(-1, 1)).sum().item()

        total += labels.size(0)

    # Calcola le metriche finali
    top1_accuracy = 100 * correct_top1 / total
    top5_accuracy = 100 * correct_top5 / total
    avg_loss = adv_loss / len(testloader)

    print(f'Accuracy of the network on adversarial images (epsilon={epsilon}): Top-1 = {top1_accuracy}%, Top-5 = {top5_accuracy}%, Loss: {avg_loss}')



import torch

def test_teacher_student_attack(teacher_net, student_net, dist_net, testloader, device, criterion, attack_type = None, save_first=False, **kwargs):
    
    print("\n ***Teacher and student adversarial test***")
    
    total = 0

    saved_flag = False

    attack_file_name = ""

    if attack_type == "fgsm":
        print("FGSM attack")
        epsilon = kwargs.get("epsilon", 0.1)
        attack_file_name = "fgsm"+str(epsilon)
    elif attack_type == "pgd":
        print("PGD attack")
        epsilon = kwargs.get("epsilon", 0.1)
        alpha = kwargs.get("alpha", 0.01)
        num_iter = kwargs.get("num_iter", 10)
        attack_file_name = "pgd"+str(epsilon)+"_"+str(alpha)+"_"+str(num_iter)
    else:
        print("Invalid attack type")
        exit(1)
    
    # Loss e conteggio delle corrette
    adv_teacher_loss = 0
    adv_student_loss = 0
    adv_dist_student_loss = 0
    teacher_loss = 0    
    student_loss = 0
    dist_student_loss = 0
    
    # Accuratezza Top-1 e Top-5
    adv_student_correct_top1 = 0
    adv_teacher_correct_top1 = 0
    adv_dist_student_correct_top1 = 0
    student_correct_top1 = 0
    teacher_correct_top1 = 0
    dist_student_correct_top1 = 0
    
    adv_student_correct_top5 = 0
    adv_teacher_correct_top5 = 0
    adv_dist_student_correct_top5 = 0
    student_correct_top5 = 0
    teacher_correct_top5 = 0
    dist_student_correct_top5 = 0

    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        

        if attack_type == "fgsm":
            adv_teacher_images = fgsm_attack(images, labels, teacher_net, epsilon, criterion)

        elif attack_type == "pgd":
            adv_teacher_images = pgd_attack(images, labels, teacher_net, epsilon, alpha, num_iter, criterion)
        
        
        # Ottieni le predizioni su immagini avversariali
        adv_teacher_outputs = teacher_net(adv_teacher_images)
        adv_student_outputs = student_net(adv_teacher_images)
        adv_dist_student_outputs = dist_net(adv_teacher_images)

        # Ottieni le predizioni su immagini normali
        teacher_outputs = teacher_net(images)
        student_outputs = student_net(images)
        dist_student_outputs = dist_net(images)

        # Calcola la loss per ogni modello
        adv_teacher_loss += criterion(adv_teacher_outputs, labels).item()
        adv_student_loss += criterion(adv_student_outputs, labels).item()
        adv_dist_student_loss += criterion(adv_dist_student_outputs, labels).item()
        teacher_loss += criterion(teacher_outputs, labels).item()
        student_loss += criterion(student_outputs, labels).item()
        dist_student_loss += criterion(dist_student_outputs, labels).item()

        # Calcolo delle predizioni corrette Top-1 e Top-5 per ciascun modello e per immagini normali e avversariali

        # Teacher (Avversariali)
        _, adv_teacher_pred_top1 = torch.max(adv_teacher_outputs, 1)
        _, adv_teacher_pred_top5 = adv_teacher_outputs.topk(5, dim=1, largest=True, sorted=True)
        adv_teacher_correct_top1 += (adv_teacher_pred_top1 == labels).sum().item()
        adv_teacher_correct_top5 += (adv_teacher_pred_top5 == labels.view(-1, 1)).sum().item()

        # Student (Avversariali)
        _, adv_student_pred_top1 = torch.max(adv_student_outputs, 1)
        _, adv_student_pred_top5 = adv_student_outputs.topk(5, dim=1, largest=True, sorted=True)
        adv_student_correct_top1 += (adv_student_pred_top1 == labels).sum().item()
        adv_student_correct_top5 += (adv_student_pred_top5 == labels.view(-1, 1)).sum().item()

        # Distilled Student (Avversariali)
        _, adv_dist_student_pred_top1 = torch.max(adv_dist_student_outputs, 1)
        _, adv_dist_student_pred_top5 = adv_dist_student_outputs.topk(5, dim=1, largest=True, sorted=True)
        adv_dist_student_correct_top1 += (adv_dist_student_pred_top1 == labels).sum().item()
        adv_dist_student_correct_top5 += (adv_dist_student_pred_top5 == labels.view(-1, 1)).sum().item()

        # Teacher (Normali)
        _, teacher_pred_top1 = torch.max(teacher_outputs, 1)
        _, teacher_pred_top5 = teacher_outputs.topk(5, dim=1, largest=True, sorted=True)
        teacher_correct_top1 += (teacher_pred_top1 == labels).sum().item()
        teacher_correct_top5 += (teacher_pred_top5 == labels.view(-1, 1)).sum().item()

        # Student (Normali)
        _, student_pred_top1 = torch.max(student_outputs, 1)
        _, student_pred_top5 = student_outputs.topk(5, dim=1, largest=True, sorted=True)
        student_correct_top1 += (student_pred_top1 == labels).sum().item()
        student_correct_top5 += (student_pred_top5 == labels.view(-1, 1)).sum().item()

        # Distilled Student (Normali)
        _, dist_student_pred_top1 = torch.max(dist_student_outputs, 1)
        _, dist_student_pred_top5 = dist_student_outputs.topk(5, dim=1, largest=True, sorted=True)
        dist_student_correct_top1 += (dist_student_pred_top1 == labels).sum().item()
        dist_student_correct_top5 += (dist_student_pred_top5 == labels.view(-1, 1)).sum().item()

        # Calcolo totale delle immagini
        total += labels.size(0)

        if save_first and not saved_flag:
            # Salva la prima immagine avversariale
            print("Saving the first adversarial image")
            combined_images = torch.cat((images, adv_teacher_images), dim=0)
            grid = torchvision.utils.make_grid(combined_images, nrow=testloader.batch_size)
            torchvision.utils.save_image(grid.to('cpu'), "work/project/adv_results/" + dataset_name + "/dist_test_image" + attack_file_name + ".png")
            saved_flag = True
            #save outputs in a file txt
            with open("work/project/saved_fig/" + attack_file_name + ".txt", "w") as f:
                f.write("Teacher outputs\n" + str(adv_teacher_outputs) + "\n")
                f.write("Student outputs\n" + str(adv_student_outputs) + "\n")
                f.write("Distilled Student outputs\n" + str(adv_dist_student_outputs) + "\n")
                f.write("Teacher normal outputs\n" + str(teacher_outputs) + "\n")
                f.write("Student normal outputs\n" + str(student_outputs) + "\n")
                f.write("Distilled Student normal outputs\n" + str(dist_student_outputs) + "\n")
                f.write("Teacher correct top1\n" + str(teacher_correct_top1) + "\n")
                f.write("Student correct top1\n" + str(student_correct_top1) + "\n")
                f.write("Distilled Student correct top1\n" + str(dist_student_correct_top1) + "\n")
                f.write("Teacher correct top5\n" + str(teacher_correct_top5) + "\n")
                f.write("Student correct top5\n" + str(student_correct_top5) + "\n")
                f.write("Distilled Student correct top5\n" + str(dist_student_correct_top5) + "\n")
                f.write("Teacher loss\n" + str(teacher_loss) + "\n")
                f.write("Student loss\n" + str(student_loss) + "\n")
                f.write("Distilled Student loss\n" + str(dist_student_loss) + "\n")


    # Stampa delle metriche finali
    print(f'Adversarial - Teacher: Top-1 Accuracy: {100 * adv_teacher_correct_top1 / total}%, Top-5 Accuracy: {100 * adv_teacher_correct_top5 / total}%, Loss: {adv_teacher_loss / len(testloader)}')
    print(f'Adversarial - Student: Top-1 Accuracy: {100 * adv_student_correct_top1 / total}%, Top-5 Accuracy: {100 * adv_student_correct_top5 / total}%, Loss: {adv_student_loss / len(testloader)}')
    print(f'Adversarial - Distilled Student: Top-1 Accuracy: {100 * adv_dist_student_correct_top1 / total}%, Top-5 Accuracy: {100 * adv_dist_student_correct_top5 / total}%, Loss: {adv_dist_student_loss / len(testloader)}')
    
    print(f'Normal - Teacher: Top-1 Accuracy: {100 * teacher_correct_top1 / total}%, Top-5 Accuracy: {100 * teacher_correct_top5 / total}%, Loss: {teacher_loss / len(testloader)}')
    print(f'Normal - Student: Top-1 Accuracy: {100 * student_correct_top1 / total}%, Top-5 Accuracy: {100 * student_correct_top5 / total}%, Loss: {student_loss / len(testloader)}')
    print(f'Normal - Distilled Student: Top-1 Accuracy: {100 * dist_student_correct_top1 / total}%, Top-5 Accuracy: {100 * dist_student_correct_top5 / total}%, Loss: {dist_student_loss / len(testloader)}')



if __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    dataset_name = "imagenette"
    dataset_path_root = './work/project/data/'
    batch_size = 64
    num_workers = 8
    student_model_name = "resnet18"
    student_model_path = 'work/project/save/imagenette/resnet18_0.0001_200/state_dict.pth'
    dist_student_model_path = 'work/project/save/imagenette/resnet18_0.0001_200_disillation_from_resnet18/state_dict.pth'
    teacher_model_name = "resnet18"
    teacher_model_path = 'work/project/save/imagenette/resnet18_0.0001_1_pretrained/state_dict.pth'



    trainloader, testloader, n_cls = get_train_and_test_loader(dataset_name, 
                                                                data_folder=dataset_path_root, 
                                                                batch_size=batch_size, 
                                                                num_workers=num_workers)
    



    print(dataset_name," - Trainloader lenght: ", len(trainloader), "Testloader lenght: ", len(testloader))


    teacher = model_dict[student_model_name](num_classes=n_cls).to(device)
    teacher.load_state_dict(torch.load(teacher_model_path, map_location=device))
    teacher.eval().to(device)

    student = model_dict[student_model_name](num_classes=n_cls).to(device)
    student.load_state_dict(torch.load(student_model_path, map_location=device))
    student.eval().to(device)

    dist_student = model_dict[student_model_name](num_classes=n_cls).to(device)
    dist_student.load_state_dict(torch.load(dist_student_model_path, map_location=device))
    dist_student.eval().to(device)





    criterion = nn.CrossEntropyLoss()

    # Esegui il test normale
    print("Testing with normal examples")
    test(teacher, testloader, criterion, device)

    # Esegui l'attacco FGSM
    epsilon = 0.1  # Modifica questo valore per aumentare o diminuire la forza dell'attacco
    alpha = 0.01
    num_iter = 20
    print("Testing with adversarial examples, epsilon=", epsilon)
    test_teacher_student_attack(teacher, student, dist_student, testloader, device, criterion, attack_type="fgsm", save_first=True, epsilon=epsilon)
    test_teacher_student_attack(teacher, student, dist_student, testloader, device, criterion, attack_type="pgd", save_first=True, epsilon=epsilon, alpha=alpha, num_iter=num_iter)

    
    epsilon = 0.3  # Modifica questo valore per aumentare o diminuire la forza dell'attacco
    alpha = 0.005
    num_iter = 100
    test_teacher_student_attack(teacher, student, dist_student, testloader, device, criterion, attack_type="pgd", save_first=True, epsilon=epsilon, alpha=alpha, num_iter=num_iter)
    test_teacher_student_attack(teacher, student, dist_student, testloader, device, criterion, attack_type="fgsm", save_first=True, epsilon=epsilon)

    epsilon = 0.5  # Modifica questo valore per aumentare o diminuire la forza dell'attacco
    alpha = 0.005
    num_iter = 100
    test_teacher_student_attack(teacher, student, dist_student, testloader, device, criterion, attack_type="pgd", save_first=True, epsilon=epsilon, alpha=alpha, num_iter=num_iter)



