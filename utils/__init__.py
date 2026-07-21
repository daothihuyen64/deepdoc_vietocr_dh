import os

from .constants import SERVICE_CONF
from . import file_utils


def conf_realpath(conf_name):
    conf_path = f"conf/{conf_name}"
    return os.path.join(file_utils.get_project_base_directory(), conf_path)


def read_config(conf_name=SERVICE_CONF):
    local_config = {}
    local_path = conf_realpath(f'local.{conf_name}')

    # load local config file
    if os.path.exists(local_path):
        local_config = file_utils.load_yaml_conf(local_path)
        if not isinstance(local_config, dict):
            raise ValueError(f'Invalid config file: "{local_path}".')

    global_config_path = conf_realpath(conf_name)
    global_config = file_utils.load_yaml_conf(global_config_path)

    if not isinstance(global_config, dict):
        raise ValueError(f'Invalid config file: "{global_config_path}".')

    global_config.update(local_config)
    return global_config
