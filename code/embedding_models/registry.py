''' Class used to register and create object from config '''
import logging


class EmbedderStore(object):
    '''
    Resolves models
    '''
    __default = None
    __store = {}

    @classmethod
    def register(cls, model_name, model):
        if not callable(getattr(model, 'get_arguments_from_configs')):
            raise ValueError("%s does not implement get_arguments_from_configs" % model_name)

        logger = logging.getLogger(__name__)
        logger.debug("Registring %s model" % model_name)
        cls.__store[model_name] = (model, model.get_arguments_from_configs)

    @classmethod
    def get(cls, model_name):
        if model_name in cls.__store:
            return cls.__store[model_name]

        raise ValueError("No model registered with name: %s" % model_name)

    @classmethod
    def get_from_configs(cls, exp_config, model_config):
        model_name = exp_config.TEXT_EMBEDDING_MODEL

        print("Initializing %s from config files" % model_name)
        model_cls, args_factory = cls.get(model_name)
        kwargs = args_factory(exp_config, model_config)

        return model_cls(**kwargs)

    @classmethod
    def get_registered(cls):
        return [(name, cls.get(name)) for name in cls.__store]

    @classmethod
    def set_default(cls, model_name):
        if model_name not in cls.__store:
            raise ValueError("Default model: %s has not been registered" % model_name)
        cls.__default = model_name

    @classmethod
    def get_default(cls):
        if cls.__default is None:
            raise RuntimeError("No default model registered")

        return cls.__default


def register(cls, name=None, default=False):
    '''
    Registerd function to Embedder store
    '''
    model_name = name if name else cls.__name__
    EmbedderStore.register(model_name, cls)
    if default:
        EmbedderStore.set_default(model_name)

    return cls


if __name__ == '__main__':
    from configurable import Configurable

    class Config:
        def __init__(self, **entries):
            self.__dict__.update(entries)

    class Foo(Configurable):
        def __init__(self, a, b, c=10):
            self.a, self.b, self.c = a, b, c

        @classmethod
        def get_arguments_from_configs(self, exp_cfg, model_cfg):
            kwargs = {
                'a': exp_cfg.a,
                'b': exp_cfg.b
            }
            kwargs.update(model_cfg)

            return kwargs

    exp_cfg = Config(**{'TEXT_EMBEDDING_MODEL': 'foo', 'a': 1, 'b': 17})
    model_cfg = {'c': 'C'}

    register(Foo, 'foo')

    foo = EmbedderStore.get_from_configs(exp_cfg, model_cfg)

    assert foo.a == exp_cfg.a
    assert foo.b == exp_cfg.b
    assert foo.c == model_cfg['c']
