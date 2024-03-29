import argparse

parser = argparse.ArgumentParser(description='Partial Associativity Experiment via DeepEMD')

parser.add_argument('--exp_name', type=str, default='experiment')

# ----------------------------
# Dataloader Options
# ----------------------------

# For SketchyScene:
# ------------------

# parser.add_argument('--root_dir', type=str, default='/vol/research/sketchcaption/datasets/sketchyscene/SketchyScene-7k',
# 	help='Enter root directory of SketchyScene')
# /home/gy/datasets/sfsd-mat/
parser.add_argument('--root_dir', type=str, default='/home/gy/datasets/SFSD',
	help='Enter root directory of SFSD Dataset')
# parser.add_argument('--root_dir', type=str, default='/vol/research/sketchcaption/datasets/photosketching')

parser.add_argument('--p_mask', type=float, default=0.0, help='Probability of an instance being masked')
parser.add_argument('--max_len', type=int, default=224, help='Max Edge length of images')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
parser.add_argument('--workers', type=int, default=12, help='Num of workers in dataloader')
# parser.add_argument('--dataset', type=str, default='SFSD', help='Enter datasets name')

opts = parser.parse_args()
