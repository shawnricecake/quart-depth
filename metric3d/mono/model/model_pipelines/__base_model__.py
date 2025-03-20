import torch
import torch.nn as nn
from mono.utils.comm import get_func


class BaseDepthModel(nn.Module):
    def __init__(self, cfg,
                 # quant:
                 quant=False, calibrate=False, quant_cfg=None,
                 **kwargs) -> None:
        super(BaseDepthModel, self).__init__()
        model_type = cfg.model.type
        self.depth_model = get_func('mono.model.model_pipelines.' + model_type)(
            cfg,
            # quant:
            quant=quant, calibrate=calibrate, quant_cfg=quant_cfg,)

    def forward(self, data):
        output = self.depth_model(**data)

        return output['prediction'], output['confidence'], output

    def inference(self, data):
        with torch.no_grad():
            pred_depth, confidence, _ = self.forward(data)
        return pred_depth, confidence