import torch
import torch.nn as nn
from mono.utils.comm import get_func

class DensePredModel(nn.Module):
    def __init__(self, cfg,
                 # quant:
                 quant=False, calibrate=False, quant_cfg=None) -> None:
        super(DensePredModel, self).__init__()

        if quant_cfg is not None:   # quant model
            self.encoder = get_func('mono.model.' + cfg.model.backbone.prefix + cfg.model.backbone.type + '_quant')(
                **cfg.model.backbone,
                # quant
                quant=quant, calibrate=calibrate, quant_cfg=quant_cfg,
            )
            self.decoder = get_func('mono.model.' + cfg.model.decode_head.prefix + cfg.model.decode_head.type + '_quant')(
                cfg,
                # quant
                quant=quant, calibrate=calibrate, quant_cfg=quant_cfg,
            )
        else:
            self.encoder = get_func('mono.model.' + cfg.model.backbone.prefix + cfg.model.backbone.type)(
                **cfg.model.backbone,
            )
            self.decoder = get_func('mono.model.' + cfg.model.decode_head.prefix + cfg.model.decode_head.type)(
                cfg,
            )

    def forward(self, input, **kwargs):
        # [f_32, f_16, f_8, f_4]
        features = self.encoder(input)
        out = self.decoder(features, **kwargs)
        return out