import torch
import torch.nn as nn
from torch.autograd import Variable

class PrunableConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.taylor_estimates = None
        self._recent_activations = None
        self._pruning_hook = None

    def forward(self, x):
        output = super().forward(x)
        self._recent_activations = output.clone()
        return output

    def set_pruning_hooks(self):
        self._pruning_hook = self.register_backward_hook(self._calculate_taylor_estimate)

    def _calculate_taylor_estimate(self, _, grad_input, grad_output):
        # skip dim 1 as it is kernel size
        estimates = self._recent_activations.mul_(grad_output[0])
        estimates = estimates.mean(dim=(0, 2, 3))        

        # normalization
        self.taylor_estimates = torch.abs(estimates) / torch.sqrt(torch.sum(estimates * estimates))
        del estimates, self._recent_activations
        self._recent_activations = None

    def prune_feature_map(self, map_index):
        is_cuda = self.weight.is_cuda

        indices = Variable(torch.LongTensor([i for i in range(self.out_channels) if i != map_index]))
        indices = indices.cuda() if is_cuda else indices

        self.weight = nn.Parameter(self.weight.index_select(0, indices).data)
        self.bias = nn.Parameter(self.bias.index_select(0, indices).data)
        self.out_channels -= 1

    def drop_input_channel(self, index):
        """
        Used when a previous conv layer is pruned. Reduces input channel count by 1
        """
        is_cuda = self.weight.is_cuda

        indices = Variable(torch.LongTensor([i for i in range(self.in_channels) if i != index]))
        indices = indices.cuda() if is_cuda else indices

        self.weight = nn.Parameter(self.weight.index_select(1, indices).data)
        self.in_channels -= 1

class PrunableBatchNorm2d(nn.BatchNorm2d):
    def drop_input_channel(self, index):
        """
        Used when a previous conv layer is pruned. Reduces input channel count by 1
        """
        if self.affine:
            is_cuda = self.weight.is_cuda
            indices = Variable(torch.LongTensor([i for i in range(self.num_features) if i != index]))
            indices = indices.cuda() if is_cuda else indices

            self.weight = nn.Parameter(self.weight.index_select(0, indices).data)
            self.bias = nn.Parameter(self.bias.index_select(0, indices).data)
            self.running_mean = self.running_mean.index_select(0, indices.data)
            self.running_var = self.running_var.index_select(0, indices.data)

        self.num_features -= 1