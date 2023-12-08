from utils.train_utils import *
from model_factory.LSDD import train_lsdd

class baseTrainer(object):

    """
    An object class for model training with self-adaptive process to select most reliable negative bags
    """

    def __init__(self,
                 config: Optional[Dict] = None,
                 model=None, bag=None):

        r"""
        Initialization function for the class

            Args:
                    config (Dict): A dictionary containing training configurations.
                    model: Model to train
                    bag: Bag dataset input

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

        self.log = self.config['save_dir'] + "/" + self.suffix + "experiment_log.txt"

    def expriment(self, idx):

        """
        Instance method to execute one experiment
        """

        self.train_bag = [self.bag.bags[item] for item in self.train_idx[idx]]
        self.val_bag = [self.bag.bags[item] for item in self.val_idx[idx]]

        bag = self.train_bag + self.val_bag

        train_bag_label = torch.stack([self.bag.labels[item].max() for item in self.train_idx[idx]]).float()
        val_bag_label = torch.stack([self.bag.labels[item].max() for item in self.val_idx[idx]]).float()
        total_label = torch.concat([train_bag_label, val_bag_label])
        n_pos = self.config['n_pos']
        pos_idx = np.random.choice(torch.where(total_label == 1)[0], size=n_pos, replace=False)
        s = torch.zeros(len(total_label))
        s[pos_idx] = 1

        clf, best_param = train_lsdd(bag, self.config)

        self.test_bag = [self.bag.bags[item] for item in self.test_idx[idx]]
        self.test_bag_label = [self.bag.labels[item] for item in self.test_idx[idx]]


    def run(self):

        "1. Split dataset: 5-fold-cross-validation"
        self.train_idx, self.val_idx, self.test_idx = SplitBag(n_splits=self.n_splits,
                                                               num_bag=self.bag.num_bag,
                                                               bag_labels=self.bag.labels)

        "2. train and save model"
        for idx in range(self.n_splits):
            self.expriment(idx)


