import os

from .constants import SERVICE_CONF
from . import file_utils


def conf_realpath(conf_name):
    conf_path = f"conf/{conf_name}"
    return os.path.join(file_utils.get_project_base_directory(), conf_path)


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge `override` into `base`, in place, returning `base`.

    A plain `base.update(override)` replaces an entire nested dict (e.g.
    `ocr:`) wholesale even if `override` only meant to set one sibling key --
    silently dropping every other key under that section (this caused a real
    incident: local.pipeline_conf.yaml's `ocr: {det_backend: mineru}` wiped
    out `ocr.fastocr_max_batch_size` from the global config, and the code
    silently fell back to its own hardcoded default instead).
    """
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


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

    return _deep_merge(global_config, local_config)
