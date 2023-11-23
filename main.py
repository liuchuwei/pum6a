import torch.utils.data as data_utils
from utils.bag_utils import Bags


# load data
train_loader = data_utils.DataLoader(Bags(dataset="annthyroid", train=True),
                                     batch_size=1,
                                     shuffle=True)

test_loader = data_utils.DataLoader(Bags(dataset="annthyroid", train=False),
                                     batch_size=1,
                                     shuffle=True)

# build model


# train & test
