"""
"""

from pprint import pprint

import matplotlib.pyplot as plt
import torch
from torch.optim.lr_scheduler import SequentialLR, LinearLR, ConstantLR, ExponentialLR


model      = torch.nn.Linear(1, 1)
optimizer  = torch.optim.Adam(model.parameters(), lr=0.001)
schedulers = [
    LinearLR(optimizer, start_factor=0.01, total_iters=2),
    ConstantLR(optimizer, factor=1.0, total_iters=3),
    ExponentialLR(optimizer, gamma=0.60),
]
milestones = [2, 5]
scheduler  = SequentialLR(optimizer, schedulers=schedulers, milestones=milestones)

lrs = []
num_epochs = 10
for epoch in range(num_epochs):
    lrs.append(optimizer.param_groups[0]['lr'])
    scheduler.step()

pprint(lrs)

# Plot the learning rate schedule
plt.plot(range(num_epochs), lrs, marker='o', linestyle='-')
plt.xticks(list(range(num_epochs)), list(range(1, num_epochs + 1)))
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.title('Learning Rate Schedule')
plt.grid()
plt.savefig("./tmp/learning_rate_schedule.png")
plt.show()
