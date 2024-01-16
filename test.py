import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader

from options import opts
from model import TripletNetwork
from dataloader import SketchyScene, SketchyCOCO, SketchyCOCO_lf, PhotoSketching, FSCOCO, SFSD

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

if __name__ == '__main__':
    dataset_transforms = transforms.Compose([
        transforms.Resize((opts.max_len, opts.max_len)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # val_dataset = SketchyScene(opts, mode='val', transform=dataset_transforms)

    # val_dataset = SketchyCOCO(opts, mode='val', transform=dataset_transforms)

    val_dataset = SFSD(opts, mode='test', transform=dataset_transforms)

    # val_dataset = FSCOCO(opts, mode='test', transform=dataset_transforms)

    # val_dataset = SFSD(opts, mode='test', transform=dataset_transforms)

    val_loader = DataLoader(
        dataset=val_dataset, batch_size=opts.batch_size, num_workers=opts.workers)

    model = TripletNetwork().load_from_checkpoint(checkpoint_path="checkpoints/SFSD/model-epoch=99-1227.ckpt")

    # model = TripletNetwork().load_from_checkpoint(checkpoint_path="checkpoints/SketchyCOCO/model-epoch=199-1227.ckpt")
    # model = TripletNetwork().load_from_checkpoint(checkpoint_path="saved_model/deepemd-photosketching-epoch=29-top10=1.00.ckpt")

    trainner=Trainer(logger=False, gpus=-1)
    trainner.test(model,val_loader)
