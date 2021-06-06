import math
import numpy as np
import paddle 
import paddle.nn as nn
import paddle.nn.functional as F

from .builder import MODELS
from ..backbones import build_backbone


@MODELS.register()
class CLIPWrapper(nn.Layer):
    def __init__(self,
                 model_name,
                 architecture,
                 minibatch_size
                 ):
        """A wrapper for a CLIP model as specified in the paper.

        Args:
            model_name (str): A case sensitive visual model name.
            architecture (dict): A dictionary containing the CLIP instantiation parameters.
        """
        super().__init__()

        self.model_name = model_name
        self.model = build_backbone(architecture) 
        self.minibatch_size = minibatch_size
        self.isViT = 'ViT' in self.model_name
        self.image_loss = nn.CrossEntropyLoss()
        self.text_loss = nn.CrossEntropyLoss()

        self.automatic_optimization = False
    
    def forward(self, image, text):
        return self.model(image, text)
    
    def get_image_logits(self, x, text_embed):
        return F.normalize(self.model.encode_image(x), dim=1) @ text_embed.t() * self.model.logit_scale.exp()

    def get_text_logits(self, x, image_embed):
        return F.normalize(self.model.encode_text(x), dim=1) @ image_embed.t() * self.model.logit_scale.exp()
    
    # Training loss: https://github.com/openai/CLIP/issues/83
    # Mini-batching thanks to https://github.com/crowsonkb / https://twitter.com/RiversHaveWings
    # Multi-GPU support: https://github.com/MicPie/clasp
    def training_step(self, train_batch, idx):
        # get optimizers and scheduler
        optimizer = self.optimizers()

        image, text = train_batch
        n = math.ceil(len(image) // self.minibatch_size) if self.minibatch_size > 0 else 1
        batch_offset = self.global_rank * len(image) # offset to align across gpus
        image_mbs = torch.chunk(image, n)
        text_mbs = torch.chunk(text, n)

        # calculate original statistics
        with torch.no_grad():
            ims = [F.normalize(self.model.encode_image(im), dim=1) for im in image_mbs]
            txt = [F.normalize(self.model.encode_text(t), dim=1) for t in text_mbs]
            # gather from all GPUs
            ims = self.all_gather(torch.cat(ims))
            txt = self.all_gather(torch.cat(txt))

            if not isinstance(ims, list):
                ims = [ims]
                txt = [txt]

            image_logits = torch.cat(ims) @ torch.cat(txt).t() * self.model.logit_scale.exp()
            ground_truth = torch.arange(len(image_logits)).type_as(image_logits).long()
            loss = (self.image_loss(image_logits, ground_truth) + self.text_loss(image_logits.t(), ground_truth)).div(2)
            acc = (torch.argmax(image_logits, 0) == ground_truth).sum()
            self.log_dict({'loss': loss, 'acc': acc / len(image)}, prog_bar=True)
        
        optimizer.zero_grad()

        # image loss
        for j, mb in enumerate(image_mbs):
            # images_tmp = ims.copy()
            image_logits = self.get_image_logits(mb, torch.cat(txt))
            ground_truth = torch.arange(len(mb)).type_as(image_logits).long() + len(mb) * j + batch_offset
            loss = self.image_loss(image_logits, ground_truth)
            self.manual_backward(loss)

        # text loss
        for j, mb in enumerate(text_mbs):
            # images_tmp = ims.copy()
            text_logits = self.get_text_logits(mb, torch.cat(ims))
            ground_truth = torch.arange(len(mb)).type_as(image_logits).long() + len(mb) * j + batch_offset
            loss = self.image_loss(text_logits, ground_truth)
            self.manual_backward(loss)
        
        optimizer.step()
        lr_scheduler = self.lr_schedulers()
        lr_scheduler.step()
        self.model.logit_scale.data.clamp_(-float('inf'), np.log(100))
    
    def validation_step(self, val_batch, idx):
        image, text = val_batch
        image_logits, text_logits = self.forward(image, text)
        ground_truth = torch.arange(len(image_logits))
        loss = (self.image_loss(image_logits, ground_truth) + self.text_loss(text_logits, ground_truth)).div(2)
        self.log('val_loss', loss)
    
    def configure_optimizers(self):
        lr = {
            "RN50": 5e-4,
            "RN101": 5e-4,
            "RN50x4": 5e-4,
            "RN50x16": 4e-4,
            "RN50x64": 3.6e-4,
            "ViT-B/32": 5e-4,
            "ViT-B/16": 5e-4,
            "ViT-L/14": 4e-4,
            "ViT-L/14-336px": 2e-5
        }[self.model_name]

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            betas=(
                0.9,
                0.98 if self.isViT else 0.999
            ),
            eps=1e-6 if self.isViT else 1e-8,
            weight_decay=0.2
        )

        # TODO Watch: https://github.com/openai/CLIP/issues/107
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=2000
        )

        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}
