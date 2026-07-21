import os

import yaml

_CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'config')

_NAME_TO_FILE = {
    'vgg_seq2seq': 'vgg-seq2seq.yml',
    'base': 'base.yml',
}


def load_config(config_file):
    with open(config_file, encoding='utf-8') as f:
        config = yaml.safe_load(f)

    return config


class Cfg(dict):
    def __init__(self, config_dict):
        super(Cfg, self).__init__(**config_dict)
        self.__dict__ = self

    @staticmethod
    def load_config_from_file(fname, base_file=None):
        base_file = base_file or os.path.join(_CONFIG_DIR, 'base.yml')
        base_config = load_config(base_file)

        with open(fname, encoding='utf-8') as f:
            config = yaml.safe_load(f)
        base_config.update(config)

        return Cfg(base_config)

    @staticmethod
    def load_config_from_name(name):
        if name not in _NAME_TO_FILE:
            raise ValueError(f"Unknown vietocr base config name: {name!r}")
        fname = os.path.join(_CONFIG_DIR, _NAME_TO_FILE[name])
        return Cfg.load_config_from_file(fname)

    def save(self, fname):
        with open(fname, 'w') as outfile:
            yaml.dump(dict(self), outfile, default_flow_style=False, allow_unicode=True)
