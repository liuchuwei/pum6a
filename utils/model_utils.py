from model_factory.PUM6A import pum6a

def LoadModel(config):

    """
    Method to load model_factory according to config

        Args:
            config (dict): A dictionary containing dataset configurations.

        Return:
            model: Positive and Unlabeled Multi-Instance Model
    """

    if config['model_chosen']=='pum6a':
        model = pum6a(config)

    elif config['model_chosen']=='puma':
        pass

    elif config['model_chosen']=='iAE':
        pass

    elif config['model_chosen']=='puIF':
        pass

    elif config['model_chosen']=='RF':
        pass