import numpy as np
import torch
import torch.utils.data as data


arr = np.array([[1,2,3],[4,5,6]])
train_dataset = data.TensorDataset(torch.from_numpy(arr))

train_loader = data.DataLoader(
        dataset=train_dataset,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        pin_memory=True
    )

for step, data_sample in enumerate(train_loader):
    inputs = data_sample
    input = inputs[0]
    print(inputs)