from cnnClassifier.constants import *
from cnnClassifier.utils.common import read_yaml, create_directories
from cnnClassifier.entity.config_entity import DataIngestionConfig
from pathlib import Path

class ConfigurationManager:
    def __init__(self, config_filepath="config/config.yaml", params_filepath="params.yaml"):
        self.config_filepath = Path(config_filepath).resolve()
        self.params_filepath = Path(params_filepath).resolve()

        # Read YAML and convert to Box for attribute-style access
        self.config = read_yaml(self.config_filepath)
        self.params = read_yaml(self.params_filepath)


        create_directories([self.config.artifacts_root])
 
    def get_data_ingestion_config(self):
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir 
        )

        return data_ingestion_config