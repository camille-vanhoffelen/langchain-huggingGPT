import logging.config
import tomllib

LOGGING_CONFIG_FILE = "logging-config.toml"


def setup_logging():
    with open("logging-config.toml", "rb") as f:
        config = tomllib.load(f)
        logging.config.dictConfig(config)
