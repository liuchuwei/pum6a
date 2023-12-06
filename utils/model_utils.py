def LoadModel(config):

    """
    Method to load model_factory according to config

        Args:
            config (dict): A dictionary containing dataset configurations.

        Return:
            model: Positive and Unlabeled Multi-Instance Model
    """

    if config['model_chosen']=='pum6a':
        model = pum6a(config['model_factory'])

    elif config['model_chosen']=='puma':
        model = puma(config['model_factory'])

    elif config['model_chosen']=='iAE':
        model = iAE(config['model_factory'])

    elif config['model_chosen']=='puIF':
        model = puIF(config['model_factory'])
        
    elif config['model_chosen']=='RF':
        model = RF(config['model_factory'])