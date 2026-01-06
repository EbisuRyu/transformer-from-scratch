import yaml
from config.model_config import ModelConfig
from config.train_config import TrainConfig


def load_config(
    train_path="config/train.yaml",
    model_path="config/model.yaml",
):
    with open(train_path) as f:
        train_dict = yaml.safe_load(f)
    with open(model_path) as f:
        model_dict = yaml.safe_load(f)

    return (
        TrainConfig(**train_dict),
        ModelConfig(**model_dict),
    )
