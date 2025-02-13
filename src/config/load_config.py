import yaml
import os


def load_config(config_path="config.yaml"):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def load_config_files(path):
    
    """"
    Load configuration files from the specified path.

    Args:
        - path (str): The path to the configuration files.

    Returns:
        - dict: The configuration dictionary.
    """
    config = load_config(path)
    model_type = config["info"]["model_type"]
    model_config_folder = config["info"]["model_config_folder"]
    model_config = load_config(os.path.join(model_config_folder, f"{model_type}.yaml"))
    return config, model_config
