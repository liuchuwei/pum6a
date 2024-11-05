from utils.train_utils import *

class nanoTrainer(object):

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
        self.model = model
        self.bag = bag
        self.suffix = genSuffix(config)

        self.device = (
            "cuda"
            if torch.cuda.is_available() and config['device'] == 'cuda'
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )

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
        opt_str = str(self.config['optimizer'])
        opt_str = opt_str.replace(":", "_")
        opt_str = opt_str.replace("'", "")
        opt_str = opt_str.replace(" ", "")
        opt_str = opt_str.replace("{", "")
        opt_str = opt_str.replace("}", "")
        opt_str = opt_str.replace(":", "_")
        opt_str = opt_str.replace(",", "_")
        self.log = (self.config['save_dir'] + "/" + self.suffix + opt_str
                     + "log.txt")

    def initNegLabel(self):

        """
        Instance method for initiating negative bag label
        """

        y_tmp = torch.clone(self.bag.trainBags.keys_mod)
        neg_idx = torch.where(y_tmp == 0)[0]
        self.n_pos = torch.sum(y_tmp).int()
        self.n_neg = self.n_pos
        y_tmp[neg_idx[torch.randperm(neg_idx.size(0))[:self.n_neg]]] = -1

        self.y_tmp = y_tmp.to(self.device)

    def refreshNegLabel(self, bag_scores):

        """
        Instance method for refreshing negative bag label

            Args:
                bag_scores: bag probability obtaining during training

            Returns:
                none
        """

        nonpos_idx = torch.where(self.bag.trainBags.keys_mod == 0)[0].to(self.device)
        sorted_idx = torch.argsort(bag_scores[nonpos_idx], dim=0)[:self.n_neg]
        self.y_tmp = torch.clone(self.bag.trainBags.keys_mod).to(self.device)
        self.y_tmp[nonpos_idx[sorted_idx]] = -1

    def GenDataloader(self):
        """
        Instance method for generating dataloader
        """

        self.train_loader = data_utils.DataLoader(self.bag.trainBags,
                                                  batch_size=self.config['batch_size'],
                                                  shuffle=True,
                                                  collate_fn=nano_collate)

    def train_epoch(self):
        r"""
        Instance method for taining one epoch
        """

        size=len(self.bag.trainBags)
        self.model.train()
        bag_scores = torch.zeros([len(self.bag.trainBags), 1]).to(self.device).float()

        for batch, (features, n_instance, bag_idx) in enumerate(self.train_loader):

            idx = [np.sum(n_instance[:it]) for it in range(1, len(n_instance) + 1)]
            bag = np.split(features, idx)[:-1]
            bag = [item.to(self.device) for item in bag]

            bag_labels = self.y_tmp[bag_idx]
            bag_labels = bag_labels.to(self.device)

            loss, data_inst = self.model.bag_forward((bag, bag_labels, n_instance))
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            bag_scores[bag_idx] = torch.stack([item[0] for item in data_inst])

            if self.config['verbose']:
                if batch % 5 == 0:
                    loss, current = loss.item(), (batch + 1) * self.config['batch_size']
                    print(f"likehood_loss_p: {loss:>7f}  "
                          f"[{current:>5d}/{size:>5d}]")

            # if int(batch) == 1985:
            #     # if batch % 5 == 0:
            #
            #     bag_auc = self.val_epoch()
            #     print("batch %s:" % batch)
            #     print(f"val bag_auc: {(100 * bag_auc):>0.1f}%")
            #
            #     ran_str = ''.join(random.sample(string.ascii_letters + string.digits, 5)) + "_"
            #     ran_str += self.suffix
            #     log = "batch,%s, bag_auc,%.3f,id,%s" % (batch, bag_auc, ran_str)
            #     with open(self.log, 'a+') as f:
            #         f.write(log + '\n')
            #     model_path = self.config['save_dir'] + "/" + ran_str + "model.pt"
            #     torch.save(self.model, model_path)


        self.scheduler.step()
        self.refreshNegLabel(bag_scores)

    def val_epoch(self):
        r"""
        Instance method for validate one epoch

        Return:
            bag_loss: likelihood loss of validation dataset
        """
        # self.model.eval()
        # bag_loss = self.model.validation(self.bag.valBags)
        #
        # return bag_loss
        self.model.eval()
        bag_pro, bag_y = self.model.decision(self.bag.valBags)
        bag_y[0] = 1
        bag_auc = roc_auc_score(bag_y.cpu().detach().numpy(), bag_pro.cpu().detach().numpy())
        return bag_auc

    def tesing(self):
        r"""
        Instance method for testing

        Return:
            bag_auc: auc score of bag
            ins_auc: auc score of instance
        """

        self.model.eval()
        bag_pro, bag_y = self.model.decision(self.bag.testBags)
        bag_y[0] = 1
        bag_auc = roc_auc_score(bag_y.cpu().detach().numpy(), bag_pro.cpu().detach().numpy())
        return bag_auc

    def train(self):

        """
        Instance method to train model
        """
        self.scheduler, self.optimizer = BuildOptimizer(params=self.model.parameters(),
                                                        config=self.config['optimizer'])

        self.initNegLabel()

        # best = 88888888
        # patience = 0
        for t in range(self.config['epochs']):
            if self.config['verbose']:

                print(f"Epoch {t + 1}\n-------------------------------")

            self.train_epoch()


            bag_auc = self.val_epoch()
            print(f"val bag_auc: {(100 * bag_auc):>0.1f}%")

            ran_str = ''.join(random.sample(string.ascii_letters + string.digits, 5)) + "_"
            ran_str += self.suffix
            model_path = self.config['save_dir'] + "/" + ran_str + "model.pt"
            torch.save(self.model, model_path)
            # cost = self.val_epoch()
            #
            # if self.config['verbose']:
            #
            #     print(f"val_bag_loss: {(cost):>0.1f}")
            #
            # if cost < best:
            #     best = cost
            #     patience = 0
            # else:
            #     patience += 1
            #
            # if patience == self.config['early_stopping']:
            #     break

        bag_auc = self.tesing()
        print(f"test bag_auc: {(100 * bag_auc):>0.1f}%")

        print("Finish Training!")



    def run(self):

        "1. Generating dataloader"
        self.GenDataloader()

        "2. Save initial"
        self.saveInit()

        "3. train and save model"
        self.train()


