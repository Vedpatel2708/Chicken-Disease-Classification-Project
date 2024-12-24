from cnnClassifier.constants import *
from cnnClassifier.utils.common import read_yaml, create_directories
from cnnClassifier.entity.config_entity import (DataIngestionConfig,PrepareBaseModelConfig,PrepareCallbacksConfig)
from pathlib import Path
import os

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
    
    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        config = self.config.prepare_base_model
        
        create_directories([config.root_dir])

        prepare_base_model_config = PrepareBaseModelConfig(
            root_dir=Path(config.root_dir),
            base_model_path=Path(config.base_model_path),
            updated_base_model_path=Path(config.updated_base_model_path),
            params_image_size=self.params.IMAGE_SIZE,
            params_learning_rate=self.params.LEARNING_RATE,
            params_include_top=self.params.INCLUDE_TOP,
            params_weights=self.params.WEIGHTS,
            params_classes=self.params.CLASSES
        )

        return prepare_base_model_config


     def get_prepare_callback_config(self):
        prepare_callbacks_config = PrepareCallbacksConfig(
            root_dir="artifacts/prepare_callbacks",
            tensorboard_root_log_dir="artifacts/prepare_callbacks/tensorboard_log_dir",
            checkpoint_model_filepath="artifacts/prepare_callbacks/checkpoint_dir/model.keras"  # Changed to .keras
        )
        



        prepare_callbacks_config = PrepareCallbacksConfig(
            tensorboard_root_log_dir="artifacts/prepare_callbacks/tensorboard_log_dir",
            checkpoint_model_filepath="artifacts/prepare_callbacks/checkpoint_dir/model.keras"  # Changed to .keras
         )


        return prepare_callback_config