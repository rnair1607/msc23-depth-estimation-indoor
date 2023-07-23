import numpy as np
import torch
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.functional.image import image_gradients

import torch.nn as nn

class Scale_invariant_loss(nn.Module):
    def __init__(self):
        super(Scale_invariant_loss, self).__init__()

    def forward(self, pred, gt):
        log_diff = np.log(pred) - np.log(gt)
        num_pixels = float(log_diff.size)
    
        if num_pixels == 0:
            return np.nan
        else:
            return np.sqrt(np.sum(np.square(log_diff)) / num_pixels - np.square(np.sum(log_diff)) / np.square(num_pixels))

class Custom_loss(nn.Module):
    def __init__(self):
        super(Scale_invariant_loss, self).__init__()


    def l1(self,pred,gt):
    
        assert(np.all(np.isfinite(pred) & np.isfinite(gt) & (pred > 0) & (gt > 0)))
        diff = pred - gt
        num_pixels = float(diff.size)
        
        if num_pixels == 0:
            return np.nan
        else:
            return np.sum(np.absolute(diff)) / num_pixels

    def forward(self, pred, gt):
        ssim = StructuralSimilarityIndexMeasure()
        ssim_loss = torch.mean(
            1  - ssim(pred,gt)
        )

        w_ssim = 1.0
        w_l1 = 0.1
        w_edges = 0.9
        

        # Edge Loss
        dy_gt, dx_gt = image_gradients(gt)
        dy_pred, dx_pred = image_gradients(pred)
        weights_x = torch.exp(torch.mean(torch.abs(dx_gt)))
        weights_y = torch.exp(torch.mean(torch.abs(dy_gt)))

        smoothness_x = dx_pred * weights_x
        smoothness_y = dy_pred * weights_y
        edges_loss = torch.mean(torch.abs(smoothness_x)) + torch.mean(torch.abs(smoothness_y))

        loss = (ssim_loss * w_ssim) + (self.l1(pred,gt) * w_l1) + (edges_loss * w_edges)

        return loss


# def l1(pred,gt):
    
#     assert(np.all(np.isfinite(pred) & np.isfinite(gt) & (pred > 0) & (gt > 0)))
#     diff = pred - gt
#     num_pixels = float(diff.size)
    
#     if num_pixels == 0:
#         return np.nan
#     else:
#         return np.sum(np.absolute(diff)) / num_pixels
    


# def mix_loss(pred,gt):
#     ssim = StructuralSimilarityIndexMeasure()
#     ssim_loss = torch.mean(
#         1  - ssim(pred,gt)
#     )

#     w_ssim = 1.0
#     w_l1 = 0.1
#     w_edges = 0.9
    

#     # Edge Loss
#     dy_gt, dx_gt = image_gradients(gt)
#     dy_pred, dx_pred = image_gradients(pred)
#     weights_x = torch.exp(torch.mean(torch.abs(dx_gt)))
#     weights_y = torch.exp(torch.mean(torch.abs(dy_gt)))

#     smoothness_x = dx_pred * weights_x
#     smoothness_y = dy_pred * weights_y
#     edges_loss = torch.mean(torch.abs(smoothness_x)) + torch.mean(torch.abs(smoothness_y))

#     loss = (ssim_loss * w_ssim) + (l1(pred,gt) * w_l1) + (edges_loss * w_edges)

#     return loss

# def scale_invariant(pred,gt):
#     # sqrt(Eq. 3)
#     assert(np.all(np.isfinite(pred) & np.isfinite(gt) & (pred > 0) & (gt > 0)))

#     # di = Di - Di*
#     log_diff = np.log(pred) - np.log(gt)
#     num_pixels = float(log_diff.size)
    
#     if num_pixels == 0:
#         return np.nan
#     else:
#         return np.sqrt(np.sum(np.square(log_diff)) / num_pixels - np.square(np.sum(log_diff)) / np.square(num_pixels))
    
