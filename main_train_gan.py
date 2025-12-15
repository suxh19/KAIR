import os.path
import math
import argparse
import random
import numpy as np
import logging
from typing import Dict, Any, Union
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

from utils import utils_logger
from utils import utils_image as util
from utils import utils_option as option
from utils.utils_dist import get_dist_info, init_dist

from data.select_dataset import define_Dataset
from models.select_model import define_Model


'''
# --------------------------------------------
# training code for GAN-based model
# --------------------------------------------
# Kai Zhang (cskaizhang@gmail.com)
# github: https://github.com/cszn/KAIR
# --------------------------------------------
'''





def main(json_path: str = 'options/train_msrresnet_gan.json') -> None:

    '''
    # ----------------------------------------
    # Step--1 (prepare opt)
    # ----------------------------------------
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default=json_path, help='Path to option JSON file.')
    parser.add_argument('--launcher', default='pytorch', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--dist', default=False)

    opt: Dict[str, Any] = option.parse(parser.parse_args().opt, is_train=True)
    opt['dist'] = parser.parse_args().dist

    # ----------------------------------------
    # distributed settings
    # ----------------------------------------
    if opt['dist']:
        init_dist('pytorch')
    opt['rank'], opt['world_size'] = get_dist_info()

    # opt is a dictionary with string keys, so we need to cast for type checking
    opt_dict: Dict[str, Any] = opt  # type: ignore
    if opt_dict.get('rank', 0) == 0:
        util.mkdirs((path for key, path in opt_dict.get('path', {}).items() if 'pretrained' not in key))

    # ----------------------------------------
    # update opt
    # ----------------------------------------
    # -->-->-->-->-->-->-->-->-->-->-->-->-->-
    init_iter_G, init_path_G = option.find_last_checkpoint(opt_dict.get('path', {}).get('models', ''), net_type='G')
    init_iter_D, init_path_D = option.find_last_checkpoint(opt_dict.get('path', {}).get('models', ''), net_type='D')
    init_iter_E, init_path_E = option.find_last_checkpoint(opt_dict.get('path', {}).get('models', ''), net_type='E')
    opt_dict['path']['pretrained_netG'] = init_path_G
    opt_dict['path']['pretrained_netD'] = init_path_D
    opt_dict['path']['pretrained_netE'] = init_path_E
    init_iter_optimizerG, init_path_optimizerG = option.find_last_checkpoint(opt_dict.get('path', {}).get('models', ''), net_type='optimizerG')
    init_iter_optimizerD, init_path_optimizerD = option.find_last_checkpoint(opt_dict.get('path', {}).get('models', ''), net_type='optimizerD')
    opt_dict['path']['pretrained_optimizerG'] = init_path_optimizerG
    opt_dict['path']['pretrained_optimizerD'] = init_path_optimizerD
    current_step = max(init_iter_G, init_iter_D, init_iter_E, init_iter_optimizerG, init_iter_optimizerD)

    # opt['path']['pretrained_netG'] = ''
    # current_step = 0
    border = opt_dict.get('scale', 1)
    # --<--<--<--<--<--<--<--<--<--<--<--<--<-

    # ----------------------------------------
    # save opt to  a '../option.json' file
    # ----------------------------------------
    if opt_dict.get('rank', 0) == 0:
        option.save(opt_dict)

    # ----------------------------------------
    # return None for missing key
    # ----------------------------------------
    opt = option.dict_to_nonedict(opt)

    # ----------------------------------------
    # configure logger
    # ----------------------------------------
    if opt_dict.get('rank', 0) == 0:
        logger_name = 'train'
        utils_logger.logger_info(logger_name, os.path.join(opt_dict.get('path', {}).get('log', ''), logger_name+'.log'))
        logger = logging.getLogger(logger_name)
        logger.info(option.dict2str(opt_dict))

    # ----------------------------------------
    # seed
    # ----------------------------------------
    seed = opt_dict.get('train', {}).get('manual_seed')
    if seed is None:
        seed = random.randint(1, 10000)
    print('Random seed: {}'.format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    '''
    # ----------------------------------------
    # Step--2 (creat dataloader)
    # ----------------------------------------
    '''

    # ----------------------------------------
    # 1) create_dataset
    # 2) creat_dataloader for train and test
    # ----------------------------------------
    for phase, dataset_opt in opt_dict.get('datasets', {}).items():
        if phase == 'train':
            train_set = define_Dataset(dataset_opt)
            train_size = int(math.ceil(len(train_set) / dataset_opt['dataloader_batch_size']))
            if opt_dict.get('rank', 0) == 0:
                logger.info('Number of train images: {:,d}, iters: {:,d}'.format(len(train_set), train_size))
            if opt_dict.get('dist', False):
                train_sampler = DistributedSampler(train_set, shuffle=dataset_opt['dataloader_shuffle'], drop_last=True, seed=seed)
                train_loader = DataLoader(train_set,
                                          batch_size=dataset_opt['dataloader_batch_size']//opt_dict.get('num_gpu', 1),
                                          shuffle=False,
                                          num_workers=dataset_opt['dataloader_num_workers']//opt_dict.get('num_gpu', 1),
                                          drop_last=True,
                                          pin_memory=True,
                                          sampler=train_sampler)
            else:
                train_loader = DataLoader(train_set,
                                          batch_size=dataset_opt['dataloader_batch_size'],
                                          shuffle=dataset_opt['dataloader_shuffle'],
                                          num_workers=dataset_opt['dataloader_num_workers'],
                                          drop_last=True,
                                          pin_memory=True)

        elif phase == 'test':
            test_set = define_Dataset(dataset_opt)
            test_loader = DataLoader(test_set, batch_size=1,
                                     shuffle=False, num_workers=1,
                                     drop_last=False, pin_memory=True)
        else:
            raise NotImplementedError("Phase [%s] is not recognized." % phase)

    '''
    # ----------------------------------------
    # Step--3 (initialize model)
    # ----------------------------------------
    '''

    model = define_Model(opt_dict)

    model.init_train()
    if opt_dict.get('rank', 0) == 0:
        logger.info(model.info_network())
        # logger.info(model.info_params())  # 注释掉冗长的参数打印

    '''
    # ----------------------------------------
    # Step--4 (main training)
    # ----------------------------------------
    '''

    for epoch in range(1000000):  # keep running
        if opt_dict.get('dist', False):
            train_sampler.set_epoch(epoch + seed)

        for i, train_data in enumerate(train_loader):

            current_step += 1



            # -------------------------------
            # 1) feed patch pairs
            # -------------------------------
            model.feed_data(train_data)

            # -------------------------------
            # 2) optimize parameters
            # -------------------------------
            model.optimize_parameters(current_step)

            # -------------------------------
            # 3) update learning rate
            # -------------------------------
            model.update_learning_rate(current_step)

            # -------------------------------
            # 4) training information
            # -------------------------------
            if current_step % opt_dict.get('train', {}).get('checkpoint_print', 200) == 0 and opt_dict.get('rank', 0) == 0:
                logs = model.current_log()  # such as loss
                message = '<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> '.format(epoch, current_step, model.current_learning_rate())
                for k, v in logs.items():  # merge log information into message
                    message += '{:s}: {:.3e} '.format(k, v)
                logger.info(message)

            # -------------------------------
            # 5) save model
            # -------------------------------
            if current_step % opt_dict.get('train', {}).get('checkpoint_save', 5000) == 0 and opt_dict.get('rank', 0) == 0:
                logger.info('Saving the model.')
                model.save(current_step)

            # -------------------------------
            # 6) testing
            # -------------------------------
            if current_step % opt_dict.get('train', {}).get('checkpoint_test', 5000) == 0 and opt_dict.get('rank', 0) == 0:

                avg_psnr = 0.0
                idx = 0

                for test_data in test_loader:
                    idx += 1
                    image_name_ext = os.path.basename(test_data['L_path'][0])
                    img_name, ext = os.path.splitext(image_name_ext)

                    img_dir = os.path.join(opt_dict.get('path', {}).get('images', ''), img_name)
                    util.mkdir(img_dir)

                    model.feed_data(test_data)
                    model.test()

                    visuals = model.current_visuals()
                    E_img = util.tensor2uint(visuals['E'])
                    H_img = util.tensor2uint(visuals['H'])

                    # -----------------------
                    # save estimated image E
                    # -----------------------
                    save_img_path = os.path.join(img_dir, '{:s}_{:d}.png'.format(img_name, current_step))
                    util.imsave(E_img, save_img_path)

                    # -----------------------
                    # calculate PSNR
                    # -----------------------
                    current_psnr = util.calculate_psnr(E_img, H_img, border=border)

                    logger.info('{:->4d}--> {:>10s} | {:<4.2f}dB'.format(idx, image_name_ext, current_psnr))

                    avg_psnr += current_psnr

                avg_psnr = avg_psnr / idx

                # testing log
                logger.info('<epoch:{:3d}, iter:{:8,d}, Average PSNR : {:<.2f}dB\n'.format(epoch, current_step, avg_psnr))

if __name__ == '__main__':
    main()
