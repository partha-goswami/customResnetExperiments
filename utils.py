import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms
import albumentations
from albumentations.pytorch.transforms import ToTensorV2
import numpy as np
from torchsummary import summary
from tqdm import tqdm
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, OneCycleLR
import seaborn as sns
import matplotlib.pyplot as plt

def get_device():
    '''
    This method returns the device in use.
    If cuda(gpu) is available it would return that, otherwise it would return cpu.
    '''
    use_cuda = torch.cuda.is_available()
    return torch.device("cuda" if use_cuda else "cpu")


def get_config_values():
    '''
    Returns the config data used
    '''

    dict_config_values = {}

    dict_config_values['class_names'] = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    dict_config_values['misclassified_count'] = 20

    dict_config_values['gradcam_target_layers'] = ["layer1","layer2","layer3"]

    dict_config_values['dropout_rate'] = 0.05
    dict_config_values['batch_size'] = 512
    dict_config_values['no_of_workers'] = 2
    dict_config_values['pin_memory'] = True
    dict_config_values['learning_rate'] = 0.001
    dict_config_values['epochs'] = 24
    dict_config_values['L1_factor'] = 0
    dict_config_values['L2_factor'] = 0.005
    dict_config_values['pct_start'] = 5/24
    dict_config_values['start_lr'] = 1e-4
    dict_config_values['end_lr'] = 1
    dict_config_values['lr_epochs'] = 25
    dict_config_values['momentum'] = 0.90
    dict_config_values['div_factor'] = 8
    dict_config_values['grad_clip'] = 0.1

    dict_config_values['lr_finder_learning_rate'] = 0.01
    dict_config_values['lr_finder_momentum'] = 0.9
    dict_config_values['lr_finder_end_lr'] = 0.02
    dict_config_values['lr_finder_num_iter'] = 200
    dict_config_values['lr_finder_max_lr'] = 0.0199

    dict_config_values['albumentation'] = {
        'padIfNeeded_min_height': 36,
        'padIfNeeded_min_width': 36,
        'padIfNeeded_probability': 1,

        'randomCrop_Height': 32,
        'randomCrop_Width': 32,

        'horizontalFlip_probability': 0.5,

        'cutout_probability': 0.5,
        'cutout_num_holes': 1,
        'cutout_max_h_size': 8,
        'cutout_max_w_size': 8
    }
    return dict_config_values


def get_cifar10_stats():
    '''
    we are calculating mean and standard deviation for each channel of the input data. We would use this in cutout fillvalue in albumentation transformation
    '''
    train_transform = transforms.Compose([transforms.ToTensor()])
    train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    return train_set.data.mean(axis=(0, 1, 2)) / 255, train_set.data.std(axis=(0, 1, 2)) / 255


def apply_albumentation(config_dict):
    '''
    Kept separate method to apply albumentation. As it returns the transformation applied in terms of dictionary.
    Our next dataset and dataloaders would use the transformation by calling this method.
    '''
    cifar10_mean, cifar10_std = get_cifar10_stats()

    train_transforms = albumentations.Compose(
        [albumentations.PadIfNeeded(min_height=config_dict['albumentation']['padIfNeeded_min_height'],
                                    min_width=config_dict['albumentation']['padIfNeeded_min_width'],
                                    p=config_dict['albumentation']['padIfNeeded_probability']),
         albumentations.RandomCrop(height=config_dict['albumentation']['randomCrop_Height'],
                                   width=config_dict['albumentation']['randomCrop_Width']),
         albumentations.HorizontalFlip(p=config_dict['albumentation']['horizontalFlip_probability']),
         albumentations.Cutout(num_holes=config_dict['albumentation']['cutout_num_holes'],
                               max_h_size=config_dict['albumentation']['cutout_max_h_size'],
                               max_w_size=config_dict['albumentation']['cutout_max_w_size'],
                               fill_value=tuple([x * 255.0 for x in cifar10_mean]),
                               p=config_dict['albumentation']['cutout_probability']),
         albumentations.Normalize(mean=cifar10_mean, std=cifar10_std, always_apply=True),
         ToTensorV2()
         ])

    test_transforms = albumentations.Compose(
        [albumentations.Normalize(mean=cifar10_mean, std=cifar10_std, always_apply=True),
         ToTensorV2()])
    return lambda img: train_transforms(image=np.array(img))["image"], lambda img: test_transforms(image=np.array(img))[
        "image"]


def get_data_loaders(config_dict):
    '''
    This method applies albumentation transforms and returns the train and test dataloaders
    : param config_dict: dictionary of config values
    '''
    train_transforms, test_transforms = apply_albumentation(config_dict)

    trainset = datasets.CIFAR10(root='./data', train=True,
                                download=True, transform=train_transforms)

    testset = datasets.CIFAR10(root='./data', train=False,
                               download=True, transform=test_transforms)

    train_loader = torch.utils.data.DataLoader(trainset,
                                               batch_size=config_dict['batch_size'],
                                               shuffle=True,
                                               num_workers=config_dict['no_of_workers'],
                                               pin_memory=config_dict['pin_memory'])
    test_loader = torch.utils.data.DataLoader(testset,
                                              batch_size=config_dict['batch_size'],
                                              shuffle=True,
                                              num_workers=config_dict['no_of_workers'],
                                              pin_memory=config_dict['pin_memory'])
    return train_loader, test_loader

def get_optimizer(config_dict, model):
    '''
    This method returns the optimizer
    :param config_dict: config dictionary
    :param model: model
    '''
    return optim.SGD(model.parameters(), lr=config_dict['lr_finder_learning_rate'],
                     momentum=config_dict['lr_finder_momentum'],weight_decay=config_dict['L2_factor'])

def get_scheduler(train_loader, config_dict, model):
    '''
    This method returns the scheduler
    :param train_loader: train loader
    :param config_dict: config dictionary
    :param model: model
    '''
    optimizer = get_optimizer(config_dict, model)
    return OneCycleLR(optimizer, max_lr=config_dict['lr_finder_max_lr'], epochs=config_dict['epochs'],
                      steps_per_epoch=len(train_loader), pct_start=config_dict['pct_start'],
                      div_factor=config_dict['div_factor']), optimizer

def identify_wrong_predictions(model):
    '''
    Identifies the wrong predictions and plots them
    :param model: model
    :return: None
    '''
    config_dict = get_config_values()
    _, test_loader = get_data_loaders(config_dict)
    device = get_device()
    class_names = config_dict['class_names']
    misclassified_max_count = config_dict['misclassified_count']
    #current_misclassified_count = 0

    wrong_images, wrong_label, correct_label = [], [], []

    with torch.no_grad():
        for data, target in test_loader:
            # if current_misclassified_count > misclassified_max_count:
            # break

            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True).squeeze()

            # if pred.eq(target.view_as(pred)) == False:
            wrong_pred = (pred.eq(target.view_as(pred)) == False)
            wrong_images.append(data[wrong_pred])
            wrong_label.append(pred[wrong_pred])
            correct_label.append(target.view_as(pred)[wrong_pred])
            wrong_predictions = list(zip(torch.cat(wrong_images), torch.cat(wrong_label), torch.cat(correct_label)))
            # current_misclassified_count += 1

        fig = plt.figure(figsize=(10, 12))
        fig.tight_layout()
        for i, (img, pred, correct) in enumerate(wrong_predictions[:misclassified_max_count]):
            img, pred, target = img.cpu().numpy(), pred.cpu(), correct.cpu()
            img = np.transpose(img, (1, 2, 0)) / 2 + 0.5
            ax = fig.add_subplot(5, 5, i + 1)
            ax.axis('off')
            ax.set_title(f'\nactual : {class_names[target.item()]}\npredicted : {class_names[pred.item()]}',
                         fontsize=10)
            ax.imshow(img)

        plt.show()
    return wrong_predictions


def plot_metrics(train_accuracy, train_losses, test_accuracy, test_losses):
    '''
    The method plots model metrics
    :param train_accuracy: train accuracy
    :param train_losses: train loss
    :param test_accuracy: test accuracy
    :param test_losses: test loss
    :return: None
    '''
    sns.set(font_scale=1)
    plt.rcParams["figure.figsize"] = (25, 6)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(np.array(test_losses), 'b', label="Validation Loss")

    ax1.set_title("Validation Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()

    ax2.plot(np.array(test_accuracy), 'b', label="Validation Accuracy")

    ax2.set_title("Validation Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()

    plt.show()

def plot_gradCAM(wrong_predictions, start = 0, end, model):
    '''
    This method plots gradCAM outputs by taking help from gradcam.py
    :param wrong_predictions: wrong predictions
    :param start: start
    :param end: end
    :param model: model
    :return: None
    '''
    config_dict = get_config_values()
    device = get_device()
    gradcam_output, probs, predicted_classes = generate_gradcam(wrong_predictions[start:end],
                                                                model, config_dict['gradcam_target_layers'], device)
    plot_gradcam(gradcam_output, config_dict['gradcam_target_layers'],
                 config_dict['class_names'], (3, 32, 32), predicted_classes, wrong_predictions[start:end])
