import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class MultiCrossEntropyLoss(nn.Module):
    def __init__(self, focal=False, weight=None, reduce=True):
        super(MultiCrossEntropyLoss, self).__init__()
        self.focal = focal
        self.weight= weight
        self.reduce = reduce

    def forward(self, input, target):
        #IN: input: unregularized logits [B, C] target: multi-hot representaiton [B, C]
        target_sum = torch.sum(target, dim=1)
        target_div = torch.where(target_sum != 0, target_sum, torch.ones_like(target_sum)).unsqueeze(1)
        target = target/target_div
        logsoftmax = nn.LogSoftmax(dim=1).to(input.device)
        if not self.focal:
            if self.weight is None:
                output = torch.sum(-target * logsoftmax(input), 1)
            else:
                output = torch.sum(-target * logsoftmax(input) /self.weight, 1)
        else:
            softmax = nn.Softmax(dim=1).to(input.device)
            p = softmax(input)
            output = torch.sum(-target * (1 - p)**2 * logsoftmax(input), 1)
            
        if self.reduce:
            return torch.mean(output)
        else:
            return output
    

def cls_loss_func(y,output, use_focal=False, weight=None, reduce=True):
    input_size=y.size()
    y = y.float().cuda()
    if weight is not None:
        weight = weight.cuda()
    loss_func = MultiCrossEntropyLoss(focal=use_focal, weight=weight, reduce=reduce)
    
    y=y.reshape(-1,y.size(-1))
    output=output.reshape(-1,output.size(-1))
    loss = loss_func(output,y)
    
    if not reduce:
        loss = loss.reshape(input_size[:-1])
    
    return loss


def regress_loss_func(y,output):
    y = y.float().cuda()
    
    #y=y.unsqueeze(-1)
    y=y.reshape(-1,y.size(-1))
    output=output.reshape(-1,output.size(-1))
    
    bgmask= y[:,1] < -1e2
    
    fg_logits = output[~bgmask]
    bg_logits = output[bgmask]
    
    fg_target = y[~bgmask]
    bg_target = y[bgmask]
    
    loss = nn.functional.l1_loss(fg_logits,fg_target)
        
    if(loss.isnan()):
        return torch.tensor([0.0], requires_grad=True).cuda()
    return loss


def suppress_loss_func(y,output):
    y = y.float().cuda()
    
    #y=y.unsqueeze(-1)
    y=y.reshape(-1,y.size(-1))
    output=output.reshape(-1,output.size(-1))
    
    loss = nn.functional.binary_cross_entropy(output,y)
        
    return loss

