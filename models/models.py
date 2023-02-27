from torch import nn
from torchvision import models


def load_model(config):
    """
    The function of loading a model by name from a configuration file
    :param config:
    :return:
    """
    arch = config.model.arch
    num_classes = config.dataset.num_of_classes
    if arch.startswith('resnet'):
        pretrained_weights = models.resnet18(pretrained=True)
        model = models.__dict__[arch](pretrained=True)
        model.load_state_dict(pretrained_weights.state_dict())
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        raise Exception('model type is not supported:', arch)
    model.to('cpu')
    return model


