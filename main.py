import toml
import torch.utils.data as data_utils
from utils.bag_utils import Bags
from model.model_factory import milpuAttention

# load data
train_loader = data_utils.DataLoader(Bags(dataset="MNIST", train=True),
                                     batch_size=10,
                                     shuffle=True)

test_loader = data_utils.DataLoader(Bags(dataset="MNIST", train=False),
                                     batch_size=10,
                                     shuffle=True)

# build model
config = toml.load("config/MNIST_milpuAttention.toml")
model = milpuAttention(config)

for batch_idx, (data, label) in enumerate(train_loader):

    bag_label = label[0]

# train & test
