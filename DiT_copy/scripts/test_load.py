import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.utils import save_image


transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
dataset = ImageFolder("/home/hice1/pganesan8/scratch/imagenet", transform=transform)
sampler = DistributedSampler(
        dataset,
        num_replicas=1,
        rank=0,
        shuffle=True,
        seed=1
    )
loader = DataLoader(
        dataset,
        batch_size=int(20),
        shuffle=False,
        sampler=sampler,
        num_workers=1,
        pin_memory=True,
        drop_last=True
    )

sampler.set_epoch(0)
x, y = next(iter(loader))
save_image(x, "x.png", nrow=4, normalize=True, value_range=(-1, 1))
print(y)
idx_to_class = dict()
for k, v in dataset.class_to_idx.items():
    idx_to_class[v] = k

for i in range(len(y)):
    print(idx_to_class[int(y[i])])