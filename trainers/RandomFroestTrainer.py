from model_factory.factory import weightnoisyor
from utils.train_utils import *
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import roc_auc_score


class RF_Trainer(object):

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
        self.saveInit()

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

        self.log = self.config['save_dir'] + "/" + self.suffix + "experiment_log.txt"

    def expriment(self, idx):

        """
        Instance method to execute one experiment
        """

        self.train_bag = [self.bag.bags[item] for item in self.train_idx[idx]]
        self.val_bag = [self.bag.bags[item] for item in self.val_idx[idx]]

        X = self.train_bag + self.val_bag

        train_bag_label = torch.stack([self.bag.labels[item].max() for item in self.train_idx[idx]]).float()
        val_bag_label = torch.stack([self.bag.labels[item].max() for item in self.val_idx[idx]]).float()
        total_label = torch.concat([train_bag_label, val_bag_label])
        n_pos = self.config['n_pos']
        pos_idx = np.random.choice(torch.where(total_label == 1)[0], size=n_pos, replace=False)
        y = torch.zeros(len(total_label))
        y[pos_idx] = 1

        Y = []
        for id, item in enumerate(y):
            Y.append(item.repeat(len(X[id])))

        X = torch.concat(X)
        Y = torch.concat(Y)

        clf = RandomizedSearchCV(estimator=self.init_model.rf,
                                       param_distributions=self.init_model.random_grid,
                                       n_iter=100, cv=3, verbose=2,
                                       random_state=666, n_jobs=-1)

        clf.fit(X, Y)

        self.test_bag = [self.bag.bags[item] for item in self.test_idx[idx]]
        self.test_bag_label = [self.bag.labels[item] for item in self.test_idx[idx]]

        ins_pro = []
        bag_pro = []
        for item in self.test_bag:

            p = clf.predict(item)
            p = torch.Tensor(p)
            ins_pro.append(p)
            p = p.unsqueeze(0)
            bag_pro.append(weightnoisyor(p))

        inst_y = torch.concat(self.test_bag_label).float()
        inst_pro = torch.concat(ins_pro).squeeze().detach().numpy()

        bag_y = torch.stack([item.max() for item in self.test_bag_label]).float()
        bag_auc = roc_auc_score(bag_y.cpu(), torch.concat(bag_pro).cpu().squeeze().detach().numpy())
        ins_auc = roc_auc_score(inst_y.cpu(), inst_pro)

        print(f"Bag_auc: {(100 * bag_auc):>0.1f}%, "
              f"Instance_auc: {(100 * ins_auc):>0.1f}%.")

        ran_str = ''.join(random.sample(string.ascii_letters + string.digits, 5)) + "_"
        ran_str += self.suffix
        log = "bag_auc,%.3f,ins_auc,%.3f,id,%s" % (bag_auc, ins_auc, ran_str)
        with open(self.log, 'a+') as f:
            f.write(log + '\n')

    def run(self):

        "1. Split dataset: 5-fold-cross-validation"
        self.train_idx, self.val_idx, self.test_idx = SplitBag(n_splits=self.n_splits,
                                                               num_bag=self.bag.num_bag,
                                                               bag_labels=self.bag.labels)

        "2 train and save model"
        for idx in range(self.n_splits):
            self.expriment(idx)
