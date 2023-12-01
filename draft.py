

class pum6a(nn.Module):

    r"""
    The positive and unlabeled multi-instance model.

    """
    def __init__(self, model_config: Dict):

        r"""
        Initialization function for the class

            Args:
                    model_config (Dict): A dictionary containing model configurations.

            Returns:
                    None
        """

        super(pum6a, self).__init__()

        self.model_config = model_config
        self.build_model()

        self.device = (
            "cuda"
            if torch.cuda.is_available() and model_config['device'] == 'cuda'
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )

    def build_model(self):

        r'''
        Instance method for building pum6a model according to config
        '''

        self.L = self.model_config['attention']['L']
        self.D = self.model_config['attention']['D']
        self.K = self.model_config['attention']['K']


        self.feature_extractor_part1 = nn.Sequential(

            # 28*28
            nn.Conv2d(1, 20, kernel_size=5),
            # 20*24*24
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            # 20*12*12
            nn.Conv2d(20, 50, kernel_size=5),
            # 50*8*8
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
            # 50*4*4
        )

        self.feature_extractor_part2 = nn.Sequential(
            nn.Linear(50 * 4 * 4, self.L),
            nn.ReLU(),
        )

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.L * self.K, 1),
            nn.Sigmoid()
        )

        self.A = torch.nn.Parameter(torch.tensor(torch.rand(1)), requires_grad=True)
        self.B = torch.nn.Parameter(torch.tensor(torch.rand(1)), requires_grad=True)

    def Attforward(self, x):

        r'''
        Instance method to get modification probability on the site level from instance features.

                Args:
                        x (torch.Tensor): A tensor representation of the instance features
                Returns:
                        Y_prob (torch.Tensor): A tensor representation the bag probability
                        A (torch.Tensor): A tensor containing attention weight of instance features

        '''
        x = x.squeeze(0)

        H = self.feature_extractor_part1(x)
        H = H.view(-1, 50 * 4 * 4)
        H = self.feature_extractor_part2(H)  # NxL

        A = self.attention(H)  # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N

        M = torch.mm(A, H)  # KxL

        Y_prob = self.classifier(M)

        return Y_prob, A


    def bag_forward(self, input):

        r'''
        Instance method to get modification probability on the bag level from instance features.

               Args:
                       input (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
                       Tensor representation of bag, bag_labels, and number of instance
               Returns:
                       loss (torch.Tensor): A tensor representation of model loss

       '''


        bag, bag_labels, n_instance = input

        data_inst = [self.Attforward(item) for item in bag]
        idx_l2 = torch.where(bag_labels != 0)[0]
        if idx_l2.shape[0] > 0:
            l2 = [data_inst[item] for item in idx_l2]
            pi = torch.stack([item[0] for item in l2])
            y = torch.stack([bag_labels[index] for index in idx_l2])
            y[torch.where(y == -1)] = 0
            loss = torch.sum(-1. * (y * torch.log(pi) + (1. - y) * torch.log(1. - pi)))
        else:
            loss = 0.01*(self.A**2+self.B**2)[0]

        return loss, data_inst
