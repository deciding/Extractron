from .extractron import Extractron


def create_model(name, hparams, args=None):
    if name == 'Extractron':
        return Extractron(hparams, args)
    else:
        raise Exception('Unknown model: ' + name)
