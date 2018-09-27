from os.path import dirname, abspath, join, exists
import os
import logging

BASE_DIR = dirname(dirname(abspath(__file__)))


def get_logger(run_name, save_log=None):
    log_dir = join(BASE_DIR, 'logs')
    if not exists(log_dir):
        os.makedirs(log_dir)

    log_filename = f'{run_name}.log'
    if save_log is None:
        log_filepath = join(log_dir, log_filename)
    else:
        log_filepath = save_log

    logger = logging.getLogger(run_name)

    if not logger.handlers:  # execute only if logger doesn't already exist
        file_handler = logging.FileHandler(log_filepath, 'w', 'utf-8')
        stream_handler = logging.StreamHandler(os.sys.stdout)

        formatter = logging.Formatter('[%(levelname)s] %(asctime)s > %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

        file_handler.setFormatter(formatter)
        stream_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)
        logger.setLevel(logging.INFO)

    return logger