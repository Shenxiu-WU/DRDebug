import torch
import torch.nn as nn
import _iou_ext

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, bboxes: torch.Tensor, gtboxes: torch.Tensor, mode: int = 0) -> torch.Tensor:
        bboxes_x1 = bboxes[:, 0].contiguous()
        bboxes_y1 = bboxes[:, 1].contiguous()
        bboxes_x2 = bboxes[:, 2].contiguous()
        bboxes_y2 = bboxes[:, 3].contiguous()
        gtboxes_x1 = gtboxes[:, 0].contiguous()
        gtboxes_y1 = gtboxes[:, 1].contiguous()
        gtboxes_x2 = gtboxes[:, 2].contiguous()
        gtboxes_y2 = gtboxes[:, 3].contiguous()

        out = _iou_ext.run_iou(
            bboxes_x1, bboxes_y1, bboxes_x2, bboxes_y2,
            gtboxes_x1, gtboxes_y1, gtboxes_x2, gtboxes_y2,
            mode,
        )
        return out.to(bboxes.dtype)
