# We build decoder from SAM (https://github.com/facebookresearch/segment-anything)
from torch import nn

from .mask_decoder import MaskDecoder
from .reason_decoder import ReasonDecoder
from .prompt_encoder import PromptEncoder
from .reason_transformer import TwoWayTransformerReason
from .transformer import TwoWayTransformer


class Reason_Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.mask_decoder = ReasonDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformerReason(
                depth=2,
                embedding_dim=512,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=512,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        )

    def forward(self, vis, txt, state):
        out1, out2, src = self.mask_decoder(vis, txt, state, False)
        return out1, out2, src


class Mask_Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.prompt_decoder = PromptEncoder(
            embed_dim=512,
            image_embedding_size=(32, 32),
            input_image_size=(512, 512),
            mask_in_chans=16,
        )
        self.mask_decoder = MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=512,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=512,
            iou_head_depth=3,
            iou_head_hidden_dim=256, )

    def forward(self, vis,points=None, boxes=None, masks=None):
        sparse_embeddings, dense_embeddings= self.prompt_decoder(points=points,boxes=boxes,masks=masks)
        out1, out2 = self.mask_decoder(vis, sparse_embeddings, dense_embeddings,False)
        return out1