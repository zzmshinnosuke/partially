import json
from tqdm import tqdm
import torch
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
from options import opts
from model import TripletNetwork
from network import VGG_Network as Encoder
import torch.nn.functional as F
from dataloader import SketchyScene, SketchyCOCO, PhotoSketching, SFSD, SFSD1, FScoco

def test(test_dataloader, model):
    with torch.no_grad():
        recall = 0
        sk_feats = []
        img_feats = [] 
        for batch in tqdm(test_dataloader):
            sk_tensor, img_tensor, _ = batch
            # Shape of feat: B x 512 x 7 x 7
            sk_feat = model.embedding_network(sk_tensor)
            img_feat = model.embedding_network(img_tensor)
            # Reduce shape of feat: B x 512 x 3 x 3
            sk_feat = F.adaptive_avg_pool2d(sk_feat, 3).squeeze()
            img_feat = F.adaptive_avg_pool2d(img_feat, 3).squeeze()
            # Reduce shape of feat: B x 512 x 9
            sk_feat.reshape(sk_feat.shape[0], 512, -1)
            img_feat.reshape(sk_feat.shape[0], 512, -1)
            # Reduce shape of feat: B x 4608
            sk_feat = sk_feat.reshape(sk_feat.shape[0], -1)
            img_feat = img_feat.reshape(sk_feat.shape[0], -1)
            # if sk_feat.shape[0] == 16:
            sk_feats.extend(sk_feat.cpu().detach().numpy())
            img_feats.extend(img_feat.cpu().detach().numpy())
        nbrs = NearestNeighbors(n_neighbors=Top_K, algorithm='brute', metric='cosine').fit(img_feats)
        distances, indices = nbrs.kneighbors(sk_feats)
        for index,indice in enumerate(indices):
            if index in indice:
                recall += 1
        print(round(recall / len(img_feats), 4))

if __name__ == '__main__':
    dataset_transforms = transforms.Compose([
        transforms.Resize((opts.max_len, opts.max_len)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_dataset = SketchyCOCO(opts, mode='test', transform=dataset_transforms) #FScoco(test) SFSD1(test) SketchyCOCO(val)

    test_loader = DataLoader(
        dataset=test_dataset, batch_size=opts.batch_size, num_workers=opts.workers)

    Top_K = 10

    '''
    checkpoints/FScoco/model-epoch=199-1225.ckpt
    checkpoints/SFSD/model-epoch=199-1225.ckpt
    checkpoints/SketchyCOCO/model-epoch=199-1227.ckpt
    '''
    model = TripletNetwork().load_from_checkpoint(checkpoint_path="checkpoints/SketchyCOCO/model-epoch=199-1227.ckpt")
    test(test_loader, model)

'''
python test_t.py --root_dir ~/datasets/SFSD
python test_t.py --root_dir ~/datasets/FScoco
python test_t.py --root_dir ~/datasets/SketchyCOCO
'''