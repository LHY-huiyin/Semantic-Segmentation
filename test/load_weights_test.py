from collections import OrderedDict

import torch

from modeling.backbone.ghostnet_rectify import GhostNet

checkpoint1 = torch.load('F:/Code/pytorch-deeplab-dualattention/test/state_dict_73.98.pth')
checkpoint2 = torch.load('F:\\Code\\pytorch-deeplab-dualattention\\run\\pascal\\deeplab-ghostnet_rectify\\model_best.pth.tar')
model_state_dict = OrderedDict()

checkpoint_len1 = len(checkpoint1)-10
checkpoint_len2 = len(checkpoint2['state_dict'])

checkpoint1_key = list(checkpoint1.keys())
checkpoint2_key = list(checkpoint2['state_dict'].keys())
for i in range(checkpoint_len1):
    # key = checkpoint1
    model_state_dict[checkpoint2_key[i][9:]] = checkpoint1[checkpoint1_key[i]]
# for j in range(checkpoint_len1,checkpoint_len2):
#     model_state_dict[checkpoint2_key[i]] = checkpoint1[checkpoint1_key]

model = GhostNet()
model.load_state_dict(model_state_dict, strict=True)

print(checkpoint2)
