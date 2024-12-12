import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from train import test
from loaders import get_train_and_test_loader, get_cifar100_dataloaders

from SimKD.models import model_dict

def fgsm_attack(image, label, model, epsilon, criterion):


    # Imposta il modello in modalità valutazione
    model.eval()
    
    # Abilita il calcolo del gradiente
    image.requires_grad = True

    
    # Esegui il forward pass
    output = model(image)  
    #get cam from model
    
    # Calcola la perdita
    loss = criterion(output, label)
    
    # Azzerare i gradienti precedenti
    model.zero_grad()
    
    # Calcola il gradiente della perdita rispetto all'immagine
    loss.backward()
    
    # Prendi il segno del gradiente
    grad = image.grad.data
    
    # Crea la nuova immagine modificata
    perturbed_image = image + epsilon * grad.sign()
    
    # Assicurati che i valori dell'immagine siano compresi tra 0 e 1
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    
    return perturbed_image



def test_with_adversarial(net, testloader, device, epsilon, criterion, save_first=False):
    correct_top1 = 0
    correct_top5 = 0
    total = 0
    saved_flag = False
    adv_loss = 0

    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        
        # Applica l'attacco FGSM
        adv_images = fgsm_attack(images, labels, net, epsilon, criterion)

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

def test_teacher_student_diststudent_fgsm(teacher_net, student_net, dist_net, testloader, device, epsilon, criterion):
    print("\n ***Teacher and student adversarial test***")
    
    total = 0
    
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
        
        # Applica l'attacco FGSM
        adv_teacher_images = fgsm_attack(images, labels, teacher_net, epsilon, criterion)
        
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
    dataset_path = './work/project/SimKD/data'
    batch_size = 64
    num_workers = 8

    trainloader, testloader, n_cls = get_train_and_test_loader(dataset_name, 
                                                                data_folder=dataset_path, 
                                                                batch_size=batch_size, 
                                                                num_workers=num_workers)
    



    print(dataset_name," - Trainloader lenght: ", len(trainloader), "Testloader lenght: ", len(testloader))


    # import net

    model_name = "resnet18"
    net = model_dict[model_name](num_classes=n_cls).to(device)

    assert net is not None, "Model not found"




    teacher = model_dict[model_name](num_classes=n_cls).to(device)
    teacher_model_path = 'work/project/SimKD/save/teachers/models/resnet56_vanilla_cifar100_trial_0/resnet56_best.pth'

    teacher.load_state_dict(torch.load(teacher_model_path, map_location=device)['model'])
    teacher.eval()

    _ , testloader = get_train_and_test_loader("cifar100", data_folder='./work/project/data', batch_size=128, num_workers=8)


    teacher.to(device)

    criterion = nn.CrossEntropyLoss()

    # Esegui il test normale
    print("Testing with normal examples")
    test(teacher, testloader, criterion, device)

    # Esegui l'attacco FGSM
    epsilon = 0.1  # Modifica questo valore per aumentare o diminuire la forza dell'attacco
    print("Testing with adversarial examples, epsilon=", epsilon)
    test_with_adversarial(teacher, testloader, device, epsilon, criterion, save_first=True)

    epsilon = 0.3  # Modifica questo valore per aumentare o diminuire la forza dell'attacco
    print("Testing with adversarial examples, epsilon=", epsilon)
    test_with_adversarial(teacher, testloader, device, epsilon, criterion)

    epsilon = 0.5  # Modifica questo valore per aumentare o diminuire la forza dell'attacco
    print("Testing with adversarial examples, epsilon=", epsilon)
    test_with_adversarial(teacher, testloader, device, epsilon, criterion)



    student = model_dict["resnet18"](num_classes=100)
    student_model_path = 'work/project/SimKD/save/teachers/models/resnet18_vanilla_cifar100_trial_0/resnet18_best.pth'
    student.load_state_dict(torch.load(student_model_path, map_location=device)['model'])
    student.eval()

    student.to(device)

    # Esegui il test normale
    print("Testing NO KD STUDENT with normal examples")
    test(student, testloader, criterion, device)

    """
    ./save/students/models/S:resnet18_T:resnet56_cifar100_attention_r:1.0_a:1.0_b:1000.0_0
    ./save/students/models/S:resnet18_T:resnet56_cifar100_srrl_r:1.0_a:1.0_b:1.0_0
    ./save/students/models/S:resnet18_T:resnet56_cifar100_simkd_r:0.0_a:0.0_b:1.0_0
    ./save/students/models/S:resnet18_T:resnet56_cifar100_hint_r:1.0_a:1.0_b:100.0_0
    ./save/students/models/S:resnet18_T:resnet56_cifar100_vid_r:1.0_a:1.0_b:1.0_0
    ./save/students/models/S:resnet18_T:resnet56_cifar100_semckd_r:1.0_a:1.0_b:400.0_0
    ./save/students/models/S:resnet18_T:resnet56_cifar100_kd_r:1.0_a:1.0_b:0.0_0
    ./save/students/models/S:resnet18_T:resnet56_cifar100_crd_r:1.0_a:1.0_b:0.8_0
    ./save/students/models/S:resnet18_T:resnet56_cifar100_similarity_r:1.0_a:1.0_b:3000.0_0
    """


    epsilon = 0.1  # Modifica questo valore per aumentare o diminuire la forza dell'attacco


    distilled_student_att = model_dict["resnet18"](num_classes=100)
    distilled_student_att_model_path = 'work/project/SimKD/save/students/models/S:resnet18_T:resnet56_cifar100_attention_r:1.0_a:1.0_b:1000.0_0/resnet18_best.pth'
    distilled_student_att.load_state_dict(torch.load(distilled_student_att_model_path, map_location=device)['model'])
    distilled_student_att.eval()
    distilled_student_att.to(device)

    print("Testing TEACHER STUDENT STUDENT with adversarial examples, ATTENTION, epsilon=", epsilon)
    test_teacher_student_diststudent_fgsm(teacher, student, distilled_student_att, testloader, device, epsilon, criterion)

    #srrl
    distilled_student_srrl = model_dict["resnet18"](num_classes=100)
    distilled_student_srrl_model_path = 'work/project/SimKD/save/students/models/S:resnet18_T:resnet56_cifar100_srrl_r:1.0_a:1.0_b:1.0_0/resnet18_best.pth'
    distilled_student_srrl.load_state_dict(torch.load(distilled_student_srrl_model_path, map_location=device)['model'])
    distilled_student_srrl.eval()
    distilled_student_srrl.to(device)

    print("Testing TEACHER STUDENT STUDENT with adversarial examples, SRRl, epsilon=", epsilon)
    test_teacher_student_diststudent_fgsm(teacher, student, distilled_student_srrl, testloader, device, epsilon, criterion)

    #simkd
    distilled_student_simkd = model_dict["resnet18"](num_classes=100)
    distilled_student_simkd_model_path = 'work/project/SimKD/save/students/models/S:resnet18_T:resnet56_cifar100_simkd_r:0.0_a:0.0_b:1.0_0/resnet18_best.pth'
    distilled_student_simkd.load_state_dict(torch.load(distilled_student_simkd_model_path, map_location=device)['model'])
    distilled_student_simkd.eval()
    distilled_student_simkd.to(device)

    print("Testing TEACHER STUDENT STUDENT with adversarial examples, SIMKD, epsilon=", epsilon)
    test_teacher_student_diststudent_fgsm(teacher, student, distilled_student_simkd, testloader, device, epsilon, criterion)

    #hint
    distilled_student_hint = model_dict["resnet18"](num_classes=100)
    distilled_student_hint_model_path = 'work/project/SimKD/save/students/models/S:resnet18_T:resnet56_cifar100_hint_r:1.0_a:1.0_b:100.0_0/resnet18_best.pth'
    distilled_student_hint.load_state_dict(torch.load(distilled_student_hint_model_path, map_location=device)['model'])
    distilled_student_hint.eval()
    distilled_student_hint.to(device)

    print("Testing TEACHER STUDENT STUDENT with adversarial examples, HINT, epsilon=", epsilon)
    test_teacher_student_diststudent_fgsm(teacher, student, distilled_student_hint, testloader, device, epsilon, criterion)

    #vid
    distilled_student_vid = model_dict["resnet18"](num_classes=100)
    distilled_student_vid_model_path = 'work/project/SimKD/save/students/models/S:resnet18_T:resnet56_cifar100_vid_r:1.0_a:1.0_b:1.0_0/resnet18_best.pth'
    distilled_student_vid.load_state_dict(torch.load(distilled_student_vid_model_path, map_location=device)['model'])
    
    distilled_student_vid.eval()
    distilled_student_vid.to(device)

    print("Testing TEACHER STUDENT STUDENT with adversarial examples, VID, epsilon=", epsilon)
    test_teacher_student_diststudent_fgsm(teacher, student, distilled_student_vid, testloader, device, epsilon, criterion)

    #semckd

    distilled_student_semckd = model_dict["resnet18"](num_classes=100)
    distilled_student_semckd_model_path = 'work/project/SimKD/save/students/models/S:resnet18_T:resnet56_cifar100_semckd_r:1.0_a:1.0_b:400.0_0/resnet18_best.pth'
    distilled_student_semckd.load_state_dict(torch.load(distilled_student_semckd_model_path, map_location=device)['model'])
    distilled_student_semckd.eval()
    distilled_student_semckd.to(device)

    print("Testing TEACHER STUDENT STUDENT with adversarial examples, SEMCKD, epsilon=", epsilon)
    test_teacher_student_diststudent_fgsm(teacher, student, distilled_student_semckd, testloader, device, epsilon, criterion)

    #kd
    distilled_student_kd = model_dict["resnet18"](num_classes=100)
    distilled_student_kd_model_path = 'work/project/SimKD/save/students/models/S:resnet18_T:resnet56_cifar100_kd_r:1.0_a:1.0_b:0.0_0/resnet18_best.pth'
    distilled_student_kd.load_state_dict(torch.load(distilled_student_kd_model_path, map_location=device)['model'])
    
    distilled_student_kd.eval()
    distilled_student_kd.to(device)

    print("Testing TEACHER STUDENT STUDENT with adversarial examples, KD, epsilon=", epsilon)
    test_teacher_student_diststudent_fgsm(teacher, student, distilled_student_kd, testloader, device, epsilon, criterion)

    #crd
    distilled_student_crd = model_dict["resnet18"](num_classes=100)
    distilled_student_crd_model_path = 'work/project/SimKD/save/students/models/S:resnet18_T:resnet56_cifar100_crd_r:1.0_a:1.0_b:0.8_0/resnet18_best.pth'
    distilled_student_crd.load_state_dict(torch.load(distilled_student_crd_model_path, map_location=device)['model'])
    
    distilled_student_crd.eval()
    distilled_student_crd.to(device)

    print("Testing TEACHER STUDENT STUDENT with adversarial examples, CRD, epsilon=", epsilon)
    test_teacher_student_diststudent_fgsm(teacher, student, distilled_student_crd, testloader, device, epsilon, criterion)

    #similarity
    distilled_student_similarity = model_dict["resnet18"](num_classes=100)
    distilled_student_similarity_model_path = 'work/project/SimKD/save/students/models/S:resnet18_T:resnet56_cifar100_similarity_r:1.0_a:1.0_b:3000.0_0/resnet18_best.pth'
    distilled_student_similarity.load_state_dict(torch.load(distilled_student_similarity_model_path, map_location=device)['model'])
    distilled_student_similarity.eval()
    distilled_student_similarity.to(device)

    print("Testing TEACHER STUDENT STUDENT with adversarial examples, SIMILARITY, epsilon=", epsilon)
    test_teacher_student_diststudent_fgsm(teacher, student, distilled_student_similarity, testloader, device, epsilon, criterion)











    



