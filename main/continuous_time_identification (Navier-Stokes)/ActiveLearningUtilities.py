import logging
import fileinput
import subprocess
import ConfigManager
import os


class NoAvailableParameters(Exception):
    pass


# Checks if nektar mesh xml file is compressed
def is_compressed(filename):
    with open(filename, 'r') as infile:
        for line in infile:
            if "COMPRESSED" in line:
                return True
    return False


def substitute_text_in_file(filename, text_to_replace, replacement_text):
    for line in fileinput.input(filename, inplace=True):
        print(line.replace(text_to_replace, replacement_text), end="")


def create_logger(logger_name):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_log_handler = logging.FileHandler('{}.txt'.format(logger_name))
    file_log_handler.setLevel(logging.INFO)
    file_log_handler.setFormatter(formatter)
    logger.addHandler(file_log_handler)

    return logger


def convert_xml_to_vtu(mesh_xml_file_name, vtu_file_name, config_root=os.getcwd()):
    config_manager = ConfigManager.ConfigManager(config_root)
    subprocess.run(['mpirun', '-np', '1',
                    config_manager.get_field_convert_exe(),
                    mesh_xml_file_name, vtu_file_name]).check_returncode()