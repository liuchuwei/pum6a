import toml
import torch.utils.data as data_utils
from utils.bag_utils import Bags
from model.model_factory import milpuAttention
import torch
import numpy as np

# load data
def inference_collate(batch):
    n_instance = [item[0].shape[0] for item in batch]
    features = torch.cat([item[0] for item in batch])
    bag_labels = [item[1][0] for item in batch]
    instance_labels = [item[1][1] for item in batch]
    return features, n_instance, bag_labels, instance_labels

def train(dataloader, model, optimizer):
    size = len(dataloader.dataset)
    model.train()

    for batch, (features, n_instance, bag_labels, instance_labels) in enumerate(dataloader):
        idx = [np.sum(n_instance[:it]) for it in range(1, len(n_instance) + 1)]
        bag = np.split(features, idx)[:-1]

        loss = model.bag_forward((bag, bag_labels, instance_labels, n_instance))

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 5 == 0:
            loss, current = loss.item(), (batch + 1) * len(bag)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for batch_idx, (features, n_instance, bag_labels, instance_labels) in enumerate(dataloader):
            idx = [np.sum(n_instance[:it]) for it in range(1, len(n_instance) + 1)]
            bag = np.split(features, idx)[:-1]
            test_loss += model.bag_forward((bag, bag_labels, instance_labels, n_instance))
    test_loss /= num_batches
    print(f"Avg loss: {test_loss:>8f} \n")



train_loader = data_utils.DataLoader(Bags(dataset="annthyroid", train=True),
                                     batch_size=16,
                                     shuffle=True,
                                     collate_fn=inference_collate)

test_loader = data_utils.DataLoader(Bags(dataset="annthyroid", train=False),
                                     batch_size=16,
                                     shuffle=True,
                                     collate_fn=inference_collate)

# build model
config = toml.load("config/MNIST_milpuAttention.toml")
model = milpuAttention(config)

epochs = 100
optimizer = torch.optim.Adam(model.parameters())

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_loader, model, optimizer)
    test(test_loader, model)
print("Done!")

for batch_idx, (features, n_instance, bag_labels, instance_labels) in enumerate(train_loader):

    idx = [np.sum(n_instance[:it]) for it in range(1, len(n_instance) + 1)]
    bag = np.split(features, idx)[:-1]

    loss = model.bag_forward((bag, bag_labels, instance_labels, n_instance))

# train & test
