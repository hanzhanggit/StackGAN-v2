from abc import ABCMeta


class abstractclassmethod(classmethod):

    __isabstractmethod__ = True

    def __init__(self, callable):
        callable.__isabstractmethod__ = True
        super(abstractclassmethod, self).__init__(callable)


class Configurable:
    __metaclass__ = ABCMeta

    @abstractclassmethod
    def get_arguments_from_configs(cls, experiment_config, model_config):
        raise NotImplementedError
