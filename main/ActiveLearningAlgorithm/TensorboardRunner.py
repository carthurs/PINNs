import TensorboardTools
import time
import ConfigManager

if __name__ == '__main__':
    config_manager = ConfigManager.ConfigManager()
    with TensorboardTools.TensorboardProcess('{}/logs'.format(config_manager.get_master_model_data_root_path()), 6006):
        while True:
            time.sleep(1)