import torch

from utils.train_utils import *
import torch.utils.data as Data

class puIF_Trainer(object):

    """
    An object class for puIF model_factory training
    """

    def __init__(self,
                 config: Optional[Dict] = None,
                 model=None, bag=None):

        r"""
        Initialization function for the class

            Args:
                    config (Dict): A dictionary containing training configurations.
                    model: Model to train
                    train_bag: Bag dataset for model_factory training
                    test_bag: Bag dataset for model_factory testing

            Returns:
                    None
        """

        self.config = config
        self.init_model = model
        self.bag = bag
        self.n_splits = config['n_splits']
        self.suffix = genSuffix(config)

    def saveInit(self):

        """
        Instance method to create output path and save initiate model
        """

        tmp_dir = self.config['save_dir'].split('/')
        cur_dir = os.getcwd()

        for item in tmp_dir:

            cur_dir = cur_dir + "/" + item

            if not os.path.exists(cur_dir):
                os.mkdir(cur_dir)

        self.init_path = self.config['save_dir'] + "/" + self.suffix + "init_model.pt"
        self.log = self.config['save_dir'] + "/" + self.suffix + "experiment_log.txt"
        torch.save(self.init_model, self.init_path)

    def train_epoch(self):
        r"""
        Instance method for taining one epoch
        """
        self.model.train()
        size = len(self.train_loader)*self.train_loader.batch_size
        loss_sum = 0
        for batch, features in enumerate(self.train_loader):

            loss = self.model.bag_forward(features)
            loss.backward()
            loss_sum += loss
            self.optimizer.step()
            self.optimizer.zero_grad()

            if batch % 5 == 0:
                loss, current = loss.item(), (batch + 1) * self.config['batch_size']
                print(f"likehood_loss_p: {loss:>7f}  "
                      f"[{current:>5d}/{size:>5d}]")

        loss_sum /= (batch+1)

        return loss_sum

    def tesing(self):
        r"""
        Instance method for testing

        Return:
            bag_auc: auc score of bag
            ins_auc: auc score of instance
        """
        ins_pro, bag_pro = self.model.decision(self.test_bag)
        ins_pro = torch.concat([item.squeeze() for item in ins_pro])

        bag_y = torch.stack([item.max() for item in self.test_bag_label]).float()
        ins_y = torch.concat([item.squeeze() for item in self.test_bag_label]).float()


        bag_auc = roc_auc_score(bag_y, torch.stack(bag_pro).cpu().detach().numpy())
        ins_auc = roc_auc_score(ins_y, ins_pro.cpu().detach().numpy())

        return bag_auc, ins_auc

    def expriment(self, idx):

        """
        Instance method to execute one experiment
        """

        self.model = torch.load(self.init_path)
        self.model.eval()

        self.train_bag = [self.bag.bags[item] for item in self.train_idx[idx]]
        self.val_bag = [self.bag.bags[item] for item in self.val_idx[idx]]

        "1.Train isolation model"
        X = self.train_bag + self.val_bag
        X = torch.concat(X)
        self.model.clf.fit(X)

        "2.Get instance class"
        y = torch.Tensor(self.model.clf.predict(X))

        torch_dataset = Data.TensorDataset(X, y)
        self.train_loader = data_utils.DataLoader(torch_dataset,
                                                  batch_size=self.config['batch_size'],
                                                  shuffle=True)

        self.test_bag = [self.bag.bags[item] for item in self.test_idx[idx]]
        self.test_bag_label = [self.bag.labels[item] for item in self.test_idx[idx]]

        self.scheduler, self.optimizer = BuildOptimizer(params=self.model.parameters(),
                                                        config=self.config['optimizer'])



        best = 88888888
        for t in range(self.config['epochs']):

            print(f"Epoch {t + 1}\n-------------------------------")
            cost = self.train_epoch()
            print(f"val_bag_loss: {(cost):>0.1f}")

            if cost < best:
                best = cost
                patience = 0
            else:
                patience += 1

            if patience == self.config['early_stopping']:
                break

        print("Finish Training!")

        bag_auc, ins_auc = self.tesing()

        print(f"bag_auc: {(100 * bag_auc):>0.1f}%, "
              f"ins_auc: {(100 * ins_auc):>0.1f}%")

        ran_str = ''.join(random.sample(string.ascii_letters + string.digits, 5)) + "_"
        ran_str += self.suffix
        log = "bag_auc,%.3f,ins_auc,%.3f,id,%s" % (bag_auc, ins_auc, ran_str)
        with open(self.log, 'a+') as f:
            f.write(log + '\n')
        model_path = self.config['save_dir'] + "/" + ran_str + "model.pt"
        torch.save(self.model, model_path)

    def run(self):

        "1. Split dataset: 5-fold-cross-validation"
        self.train_idx, self.val_idx, self.test_idx = SplitBag(n_splits=self.n_splits,
                                                               bag_labels=self.bag.labels)
        "2. Save initial model"
        self.saveInit()

        "3. train and save model"
        for idx in range(self.n_splits):
            self.expriment(idx)
