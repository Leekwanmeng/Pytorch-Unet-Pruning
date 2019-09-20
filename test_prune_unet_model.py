import torch

# rig gradients, set all to 1, except first module's first map
pConv2ds = [module for module in module_list if issubclass(type(module), PrunableConv2d) and module._pruning]
for idx, pConv2d in enumerate(pConv2ds):
    pConv2d.taylor_estimates = torch.ones(pConv2d.taylor_estimates.size())
    if idx == 0:
        pConv2d.taylor_estimates[0] = 0.1