import logging


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