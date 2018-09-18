from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import torch.nn.functional as F
#from IPython import embed
import numpy as np

def to_contiguous(tensor):
    if tensor.is_contiguous():
        return tensor
    else:
        return tensor.contiguous()


class LanguageModelCriterion(nn.Module):
    def __init__(self):
        super(LanguageModelCriterion, self).__init__()

    def forward(self, input, target, mask=None):
        # TODO: use real mask
        # truncate to the same size
        target = target[:, :input.size(1)]
        mask = mask[:, :input.size(1)]
        input = to_contiguous(input)
        #print('input:', type(input), input.size())
        #print('target:', type(target), target.size())
        input = input.view(-1, input.size(2))
        # TODO: check following input
        #input = input.view(-1, input.size(1))
        target = to_contiguous(target).view(-1, 1)
        mask = to_contiguous(mask).view(-1, 1)
        output = - input.gather(1, target) * mask
        output = torch.sum(output) / torch.sum(mask)
        #output = - input.gather(1, target)
        #output = torch.sum(output)
        # WARNING: RENJ Modify
        #output = torch.sum(output) / torch.sum((mask > 0).float())

        return output

