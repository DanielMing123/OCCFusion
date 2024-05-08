import torch
file_path = '/workdir/OCCFusion/work_dirs/OccFusion/epoch_6.pth'
model = torch.load(file_path, map_location='cpu')
all = 0
for key in list(model['state_dict'].keys()):
    all += model['state_dict'][key].nelement()
print(all)

