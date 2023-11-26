import toml
import torch.utils.data as data_utils
from utils.bag_utils import Bags
from model.model_factory import milpuAttention
import torch
import numpy as np

# load data
def inference_collate(batch):
    n_instance = torch.LongTensor([item[0].shape[0] for item in batch])
    features = torch.cat([item[0] for item in batch])
    bag_labels = [item[1][0] for item in batch]
    instance_labels = [item[1][1] for item in batch]
    return features, n_instance, bag_labels, instance_labels

train_loader = data_utils.DataLoader(Bags(dataset="MNIST", train=True),
                                     batch_size=10,
                                     shuffle=True,
                                     collate_fn=inference_collate)

test_loader = data_utils.DataLoader(Bags(dataset="MNIST", train=False),
                                     batch_size=10,
                                     shuffle=True,
                                     collate_fn=inference_collate)

# build model
config = toml.load("config/MNIST_milpuAttention.toml")
model = milpuAttention(config)

for batch_idx, (features, n_instance, bag_labels, instance_labels) in enumerate(train_loader):

    idx = [np.sum(n_instance.numpy()[:it]) for it in range(1, len(n_instance.numpy()) + 1)]
    bag = np.split(features, idx)[:-1]

    model.bag_forward((bag, bag_labels))


# train & test
