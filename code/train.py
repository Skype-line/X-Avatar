import hydra
from omegaconf import OmegaConf
import yaml
import os
import glob
import numpy as np

import torch
import pytorch_lightning as pl

from lib.xavatar_scan_model import XAVATARSCANModel
from lib.xavatar_rgbd_model import XAVATARRGBDModel


@hydra.main(config_path="config", config_name="config")
def main(opt):

    print(OmegaConf.to_yaml(opt))

    pl.seed_everything(42, workers=True)

    torch.set_num_threads(10)

    # dataset
    datamodule = hydra.utils.instantiate(opt.datamodule.dataloader,
                                         opt.datamodule.dataloader,
                                         model_type=opt.experiments.model.model_type)
    datamodule.setup(stage='fit')
    np.savez('meta_info.npz', **datamodule.meta_info)

    data_processor = None
    if 'processor' in opt.datamodule:
        data_processor = hydra.utils.instantiate(
            opt.datamodule.processor,
            opt.datamodule.processor,
            meta_info=datamodule.meta_info,
            model_type=opt.experiments.model.model_type)

    # logger
    with open('.hydra/config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    logger = pl.loggers.WandbLogger(project='XAvatar', config=config)

    # checkpoint
    if opt.experiments.epoch == 'last':
        checkpoint_path = './checkpoints/last.ckpt'
    else:
        checkpoint_path = glob.glob('./checkpoints/epoch=%d*.ckpt' %
                                    opt.experiments.epoch)[0]
    if not os.path.exists(checkpoint_path) or not opt.experiments.resume:
        checkpoint_path = None

    checkpoint_callback = pl.callbacks.ModelCheckpoint(save_top_k=-1,
                                                       monitor=None,
                                                       dirpath='./checkpoints',
                                                       save_last=True,
                                                       every_n_val_epochs=1)

    trainer = pl.Trainer(logger=logger,
                         callbacks=[checkpoint_callback],
                         accelerator=None,
                         resume_from_checkpoint=checkpoint_path,
                         num_sanity_val_steps=0,
                         **opt.experiments.trainer)

    model_class = XAVATARSCANModel if datamodule.meta_info['representation'] == 'occ' else XAVATARRGBDModel

    if opt.experiments.load_init:
        init_model = torch.load(hydra.utils.to_absolute_path('init_model/{}_init_{}.pth'.format(
            datamodule.meta_info['representation'], datamodule.meta_info['gender'])))
        model = model_class(opt=opt.experiments.model,
                                meta_info=datamodule.meta_info,
                                data_processor=data_processor)
        model.shape_net.load_state_dict(
            init_model['shapenet_state_dict'])
        model.deformer.load_state_dict(init_model['deformer_state_dict'])
    else:
        model = model_class(
            opt=opt.experiments.model,
            meta_info=datamodule.meta_info,
            data_processor=data_processor)

    trainer.fit(model, datamodule=datamodule)


if __name__ == '__main__':
    main()