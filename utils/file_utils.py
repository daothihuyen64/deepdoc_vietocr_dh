import os

from ruamel.yaml import YAML


def get_project_base_directory(*args):
    base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
    if args:
        return os.path.join(base_dir, *args)
    return base_dir


def load_yaml_conf(conf_path):
    if not os.path.isabs(conf_path):
        conf_path = os.path.join(get_project_base_directory(), conf_path)
    try:
        with open(conf_path) as f:
            yaml = YAML(typ='safe', pure=True)
            return yaml.load(f)
    except Exception as e:
        raise EnvironmentError(
            "loading yaml file config from {} failed:".format(conf_path), e
        )
