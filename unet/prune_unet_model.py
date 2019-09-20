# full assembly of the sub-parts to form the complete net

import torch.nn.functional as F
from .prune_unet_parts import *
from operator import itemgetter

class PruneUNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(PruneUNet, self).__init__()
        self.inc = p_inconv(n_channels, 64)
        self.down1 = p_down(64, 128)
        self.down2 = p_down(128, 256)
        self.down3 = p_down(256, 512)
        self.down4 = p_down(512, 512)
        self.up1 = p_up(1024, 256)
        self.up2 = p_up(512, 128)
        self.up3 = p_up(256, 64)
        self.up4 = p_up(128, 64)
        self.outc = p_outconv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        print("x5: {}, x4: {}".format(x5.size(), x4.size()))
        x = self.up1(x5, x4)
        print("x: {}, x3: {}".format(x.size(), x3.size()))
        x = self.up2(x, x3)
        print("x: {}, x2: {}".format(x.size(), x2.size()))
        x = self.up3(x, x2)
        print("x: {}, x1: {}".format(x.size(), x1.size()))
        x = self.up4(x, x1)
        x = self.outc(x)
        return F.sigmoid(x)

    def set_pruning(self):
        prunable_modules = [module for module in self.modules()
                             if getattr(module, "prune_feature_map", False)
                             and module.out_channels > 1]

        # # Print getting all conv layers
        # for i, m in enumerate(self.modules()):
        #     if getattr(m, "prune_feature_map", False) and m.out_channels > 1:
        #         print(i, "->", m)

        for p in prunable_modules:
            p.set_pruning_hooks()
    
    def prune(self):
        # Get all layers excluding larger blocks
        module_list = [module for module in self.modules() 
                        if not isinstance(module, 
                            (nn.Sequential, 
                            p_double_conv, 
                            p_inconv, 
                            p_down, 
                            p_up, 
                            p_outconv,
                            PruneUNet)
                            )
                        ]
        # for i, module in enumerate(module_list):
        #     print(i, module)
        
        taylor_estimates_by_module = \
            [(module.taylor_estimates, idx) for idx, module in enumerate(module_list)
            if getattr(module, "prune_feature_map", False) and module.out_channels > 1]
        
        taylor_estimates_by_feature_map = \
            [(estimate, f_map_idx, module_idx)
             for estimates_by_f_map, module_idx in taylor_estimates_by_module
             for f_map_idx, estimate in enumerate(estimates_by_f_map)]

        min_estimate, min_f_map_idx, min_module_idx = min(taylor_estimates_by_feature_map, key=itemgetter(0))

        p_conv = module_list[min_module_idx]
        p_conv.prune_feature_map(min_f_map_idx)
        print("Pruned conv layer number {}, {}".format(min_module_idx, p_conv))

        # Find next conv layer to drop input channel
        is_last_conv = len(module_list)-1 == min_module_idx
        if not is_last_conv:
            p_batchnorm = module_list[min_module_idx+1]
            p_batchnorm.drop_input_channel(min_f_map_idx)
            
            next_conv_idx = min_module_idx + 2
            while next_conv_idx < len(module_list):
                if isinstance(module_list[next_conv_idx], PrunableConv2d):
                    module_list[next_conv_idx].drop_input_channel(min_f_map_idx)
                    print("Found next conv layer at number {}, {}".format(next_conv_idx, module_list[next_conv_idx]))
                    break
                next_conv_idx += 1

