import time
import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# try:
from network import VGG_Network as Encoder
from loss import QP_solver
# except:
# 	from src.deepemd.network import VGG_Network as Encoder
# 	from src.deepemd.loss import EMD_similarity, QP_solver, opencv_Solver
import pytorch_lightning as pl

distance_fn = lambda x, y: QP_solver(x, y)

class TripletNetwork(pl.LightningModule):

	def __init__(self, arch='vgg-16'):
		super().__init__()
		self.embedding_network = Encoder() # using FCN network
		# self.loss = nn.TripletMarginLoss(margin=1.0, p=2.0)
		
		# self.distance_fn = lambda x, y: 1.0 - F.cosine_similarity(x, y)
		self.distance_fn = lambda x, y: QP_solver(x, y)

		self.loss = nn.TripletMarginWithDistanceLoss(
			distance_function=self.distance_fn, margin=0.2)

	def forward(self, x):
		feature = self.embedding_network(x)
		return feature

	def configure_optimizers(self):
		optimizer = torch.optim.Adam(self.embedding_network.parameters(), lr=1e-4)
		return optimizer

	def training_step(self, batch, batch_idx):
		# defines the train loop
		sk_tensor, img_tensor, neg_tensor = batch

		# Shape of feat: B x 512 x 7 x 7
		sk_feat = self.embedding_network(sk_tensor)
		img_feat = self.embedding_network(img_tensor)
		neg_feat = self.embedding_network(neg_tensor)

		# Reduce shape of feat: B x 512 x 3 x 3
		sk_feat = F.adaptive_avg_pool2d(sk_feat, 3).squeeze()
		img_feat = F.adaptive_avg_pool2d(img_feat, 3).squeeze()
		neg_feat = F.adaptive_avg_pool2d(neg_feat, 3).squeeze()

		nbatch = sk_tensor.shape[0]

		# loss = self.loss(sk_feat, img_feat, neg_feat)
		loss = self.loss(
			sk_feat.reshape(nbatch, 512, -1),
			img_feat.reshape(nbatch, 512, -1),
			neg_feat.reshape(nbatch, 512, -1))

		self.log('train_loss', loss)
		return loss
	def test_step(self, batch, batch_idx):
		# defines the train loop
		sk_tensor, img_tensor, neg_tensor = batch

		# Shape of feat: B x 512 x 7 x 7
		sk_feat = self.embedding_network(sk_tensor)
		img_feat = self.embedding_network(img_tensor)

		# Reduce shape of feat: B x 512 x 3 x 3
		sk_feat = F.adaptive_avg_pool2d(sk_feat, 5).squeeze()
		img_feat = F.adaptive_avg_pool2d(img_feat, 5).squeeze()

		# _, flow = QP_solver(sk_feat, img_feat)
		return sk_feat, img_feat

	def test_epoch_end(self, test_step_outputs) -> None:
		Len = len(test_step_outputs)

		""" sketch_feature_all: B x dim x 7 x 7
		image_feature_all: B x dim x 7 x 7
		"""
		start_time = time.time()
		sketch_feature_all = torch.cat([test_step_outputs[i][0] for i in range(Len)]) # shape: B x dim x 7 x 7
		image_feature_all = torch.cat([test_step_outputs[i][1] for i in range(Len)]) # shape: B x dim x 7 x 7

		B, dim, _, _ = sketch_feature_all.shape
		sketch_feature_all = sketch_feature_all.reshape(B, dim, -1)
		image_feature_all = image_feature_all.reshape(B, dim, -1)

		rank = []
		for idx in tqdm.tqdm(range(sketch_feature_all.shape[0])):
		# for idx in tqdm.tqdm(random_idx):
			sketch_feature = sketch_feature_all[idx]
			distance = torch.zeros((image_feature_all.shape[0])).cuda()
			batch_size = 8
			for batch_idx in range(0, image_feature_all.shape[0], batch_size):
				img_feat = image_feature_all[batch_idx : batch_idx+batch_size]				
				distance[batch_idx : batch_idx+batch_size] = self.distance_fn(sketch_feature.unsqueeze(0).repeat(img_feat.shape[0], 1, 1), img_feat)
			# for batch_idx in range(0, image_feature_all.shape[0]):
			# 	img_feat = image_feature_all[batch_idx]
			# 	distance[batch_idx] = self.distance_fn(sketch_feature.unsqueeze(0),img_feat.unsqueeze(0))
			assert len(distance.size()) == 1, ValueError('Check dimensions of sketch_feature')

			gt_img_feat = image_feature_all[idx]
			target_distance_1 = self.distance_fn(sketch_feature.unsqueeze(0),
				gt_img_feat.unsqueeze(0))
			
			target_distance = distance[idx]

			print(distance.shape)
			print(target_distance, target_distance_1)
			# rank[idx] = distance.le(target_distance).sum()
			rank.append(distance.le(target_distance).sum())

		rank = sorted(rank)#[:252] # SketchyScene paper use only 252 of validation sample with highest IOU
		rank = torch.tensor(rank, dtype=torch.float32)
		print ('rank shape: ', rank.shape)
		top1 = rank.le(1).sum().numpy() / rank.shape[0]
		top5 = rank.le(5).sum().numpy() / rank.shape[0]
		top10 = rank.le(10).sum().numpy() / rank.shape[0]
		meanK = rank.mean().numpy()
		meanK = meanK.mean()

		print ('Time taken: %.2f | Metrics: top1 %.5f top5 %.5f top10 %.5f meanK %.2f'%(
			time.time() - start_time, top1, top5, top10, meanK))

		return top1, top5, top10, meanK