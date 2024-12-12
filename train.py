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


def get_parser():

    parser = argparse.ArgumentParser(description='Train a model on a dataset')

    parser.add_argument('--model', type=str, default="resnet18", help='Model name')
    parser.add_argument('--dataset', type=str, default="cifar10", help='Dataset name')
    parser.add_argument('--data_folder', type=str, default='./work/project/data', help='Path to dataset folder')
    parser.add_argument('--model_path', type=str, default=None, help='Path to model weights')
    parser.add_argument('--save_path_root', type=str, default='work/project/save/', help='Path to save model and logs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of workers for dataloader')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--save_path', type=str, default=None, help='Path to save model and logs')
    parser.add_argument('--device', type=str, default="cuda:0", help='Device to use (cpu or cuda:0)')
    parser.add_argument('--pretrained', action='store_true', help='Load pretrained model')
    parser.add_argument('--ensemble', action='store_true', help='Ensemble of models')
    parser.add_argument('--n_of_models', type=int, default=3, help='Number of models to ensemble')
    parser.add_argument('--distillation', action='store_true', help='Distillation flag')
    parser.add_argument('--teacher_model_name', type=str, default=None, help='Teacher model name')
    parser.add_argument('--teacher_path', type=str, default='work/project/save/imagenette/resnet18_0.0001_200_pretrained/state_dict.pth',
                         help='Teacher model path')

    return parser


if __name__ == "__main__":

    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    logger = logging.getLogger()
    parser = get_parser()
    args = parser.parse_args()   

    # Setup CUDA
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Model configuration
    model_name = args.model
    dataset_name = args.dataset
    dataset_path = args.data_folder
    batch_size = args.batch_size
    num_workers = args.num_workers
    save_path_root = args.save_path_root
    lr = args.lr
    epochs = args.epochs
    pretrained_flag = args.pretrained
    ensemble_flag = args.ensemble
    n_of_models = args.n_of_models
    distillation_flag = args.distillation
    teacher_path = args.teacher_path
    teacher_model_name = args.teacher_model_name


    try:
        trainloader, testloader, n_cls = get_train_and_test_loader(dataset_name, 
                                                               data_folder=dataset_path, 
                                                               batch_size=batch_size, 
                                                               num_workers=num_workers)

        logger.info(f"{dataset_name} - Trainloader length: {len(trainloader)}, Testloader length: {len(testloader)}")
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        exit(1)


    try:
        if ensemble_flag:
            assert n_of_models > 1, "Ensemble requires at least 2 models"
            net = ensemble_of_models(model_name=model_name, model_dict=model_dict, num_classes=n_cls, pretrained=pretrained_flag, n_of_models=n_of_models).to(device)
            assert net is not None, "Model not found"
            logger.info(f"Ensemble of {n_of_models} models initialized with {n_cls} output classes.")
        else:
            net = model_dict[model_name](num_classes=n_cls, pretrained=pretrained_flag).to(device)
            assert net is not None, "Model not found"
            logger.info(f"Model {model_name} initialized with {n_cls} output classes.")
    except Exception as e:
        logger.error(f"Error initializing model {model_name}: {e}")
        exit(1)


    if distillation_flag:
        assert teacher_model_name is not None, "Teacher model name not provided"
        assert teacher_path is not None, "Teacher path not provided"
        teacher = model_dict[teacher_model_name](num_classes=n_cls, pretrained=pretrained_flag).to(device)
        teacher.load_state_dict(torch.load(teacher_path, map_location=device))
        teacher.eval()
        logger.info(f"Teacher model loaded from {teacher_path}")


    # Training phase
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)

    #save path for model and logs
    save_path = os.path.join(save_path_root, dataset_name, model_name+"_"+str(lr)+"_"+str(epochs))

    if ensemble_flag:
        save_path = save_path + "_ensemble" + str(n_of_models)
        
    if pretrained_flag:
        save_path = save_path + "_pretrained"

    if distillation_flag:
        save_path = save_path + "_disillation_from_" + teacher_model_name

    os.makedirs(save_path, exist_ok=True)

    logger.info("Starting training...")

    try:
        if distillation_flag:
            teacher_metrics = test(teacher, testloader, criterion, device)
            logger.info(f"Teacher metrics: {teacher_metrics}")
            temperature = 3
            alpha = 0.5
            train_metrics = train_dist(net, teacher, trainloader, testloader, criterion, optimizer, device, 
                                       epochs=epochs, save_path=save_path, temperature=temperature, alpha=alpha)
        else:
            train_metrics = train(net, trainloader, testloader, criterion, optimizer, device, epochs=epochs, save_path=save_path)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        exit(1)

    # Testing phase
    logger.info("Starting testing...")


    try:
        test_metrics = test(net, testloader, criterion, device)
        logger.info(f"Test metrics: {test_metrics}")
    except Exception as e:
        logger.error(f"Testing failed: {e}")
        exit(1)


    # Saving model and logs
    # torch.save(net.state_dict(), os.path.join(save_path, 'state_dict.pth'))
    # logger.info(f"Model weights saved to {save_path}/state_dict.pth")
    #save test metrics and then training metrics

    try:
        with open(os.path.join(save_path, "test_metrics.txt"), "w") as f:
            f.write(str(test_metrics))
            f.write("\n")
            if distillation_flag:
                f.write(str(teacher_metrics))
                f.write("\n")
            #write args
            f.write(str(args))
            f.write("\n\n\n\n\n\n\n\n\n\n\n\n")
            f.write(str(train_metrics))
            logger.info(f"Test metrics and training metrics saved to {save_path}/test_metrics.txt")
    except Exception as e:
        logger.error(f"Error saving test metrics: {e}", exc_info=True)


    logger.info("Training and testing completed.")

    



