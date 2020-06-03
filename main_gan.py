import os
import logging

from rdkit import RDLogger

from args import get_GAN_config
from util_dir.utils_io import get_date_postfix

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Remove flooding logs.
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

from solver_gan import Solver
from torch.backends import cudnn


def main(config):
    # For fast training.
    cudnn.benchmark = True

    # Timestamp
    if config.mode == 'train':
        config.saving_dir = os.path.join(config.saving_dir, get_date_postfix())
        config.log_dir_path = os.path.join(config.saving_dir, 'log_dir')
        config.model_dir_path = os.path.join(config.saving_dir, 'model_dir')
        config.img_dir_path = os.path.join(config.saving_dir, 'img_dir')
    else:
        a_test_time = get_date_postfix()
        config.saving_dir = os.path.join(config.saving_dir)
        config.log_dir_path = os.path.join(config.saving_dir, 'post_test', a_test_time, 'log_dir')
        config.model_dir_path = os.path.join(config.saving_dir, 'model_dir')
        config.img_dir_path = os.path.join(config.saving_dir, 'post_test', a_test_time, 'img_dir')

    # Create directories if not exist.
    if not os.path.exists(config.log_dir_path):
        os.makedirs(config.log_dir_path)
    if not os.path.exists(config.model_dir_path):
        os.makedirs(config.model_dir_path)
    if not os.path.exists(config.img_dir_path):
        os.makedirs(config.img_dir_path)

    # Logger
    if config.mode == 'train':
        log_p_name = os.path.join(config.log_dir_path, get_date_postfix() + '_logger.log')
        logging.basicConfig(filename=log_p_name, level=logging.INFO)
        logging.info(config)

    # Solver for training and testing StarGAN.
    if config.mode == 'train':
        solver = Solver(config, logging)
    elif config.mode == 'test':
        solver = Solver(config)
    else:
        raise NotImplementedError

    solver.train_and_validate()


if __name__ == '__main__':
    config = get_GAN_config()

    print(config)
    main(config)
