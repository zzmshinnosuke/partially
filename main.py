import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader

from options import opts
from model import TripletNetwork
from dataloader import SketchyScene, SketchyCOCO, PhotoSketching, SketchyCOCO_lf, SFSD, FSCOCO
from torch.optim import Adam
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

AVAIL_GPUS = min(1, torch.cuda.device_count())

if __name__ == '__main__':
    dataset_transforms = transforms.Compose([
        transforms.Resize((opts.max_len, opts.max_len)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = SFSD(opts, mode='train', transform=dataset_transforms) #SFSD1 FScoco SketchyCOCO

    train_loader = DataLoader(
        dataset=train_dataset, batch_size=opts.batch_size, num_workers=opts.workers)
    
    model = TripletNetwork()

    logger = TensorBoardLogger("partial_logs", name="SFSD_data_1227")
    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints/SFSD',  # 保存检查点的目录
        filename='model-{epoch:02d}-1227',  # 文件名格式
        every_n_epochs = 50,
        save_weights_only=True,  # 仅保存模型的权重
        save_top_k=-1,  # 保存最好的 3 个检查点
        # mode='min'  # 以最小验证损失为目标
    )

    trainer = Trainer(gpus=AVAIL_GPUS, 
                benchmark=True,
                check_val_every_n_epoch=1, 
                max_epochs=100,
                min_epochs=0,
                logger=logger,
                callbacks=checkpoint_callback
                )
    
    # # 检查点文件路径
    # checkpoint_path = "checkpoints/model-epoch=49-1224.ckpt"

    # # 检查是否存在检查点文件
    # checkpoint = torch.load(checkpoint_path) if torch.cuda.is_available() else torch.load(checkpoint_path, map_location='cpu')

    # # 从检查点中加载模型和优化器状态
    # model.load_state_dict(checkpoint['model_state_dict'])
    # epoch = checkpoint['epoch']
    # optimizer = Adam(model.parameters(), lr=0.0001)  # 初始学习率为 0.0001
    # # 继续训练
    # trainer.fit(model, train_loader, optimizer=optimizer, start_epoch=epoch+1)

    trainer.fit(model, train_loader)
    # checkpoint_callback.best_model_path

'''
python main.py --root_dir ~/datasets/SFSD-open
python main.py --root_dir ~/datasets/fscoco
'''