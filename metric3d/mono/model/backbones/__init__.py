from .ConvNeXt import convnext_xlarge
from .ConvNeXt import convnext_small
from .ConvNeXt import convnext_base
from .ConvNeXt import convnext_large
from .ConvNeXt import convnext_tiny
from .ViT_DINO import vit_large
from .ViT_DINO_reg import vit_small_reg, vit_large_reg, vit_giant2_reg
from .ViT_DINO_reg_quant import vit_small_reg as vit_small_reg_quant, vit_large_reg as vit_large_reg_quant, vit_giant2_reg as vit_giant2_reg_quant

__all__ = [
    'convnext_xlarge', 'convnext_small', 'convnext_base', 'convnext_large', 'convnext_tiny', 'vit_small_reg', 'vit_large_reg', 'vit_giant2_reg',
    'vit_small_reg_quant', 'vit_large_reg_quant', 'vit_giant2_reg_quant'
]
