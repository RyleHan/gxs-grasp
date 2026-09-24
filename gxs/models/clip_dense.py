"""Frozen CLIP ViT-B/16 as a dense (per-patch) feature extractor.

Taking the last layer's patch tokens directly gives maps that are *anti*-
correlated with the text (on our test scenes: 26% object-selection accuracy vs
50% chance). Following MaskCLIP-style training-free tricks, we instead take the
last block's value projection only -- no query-key mixing, no residual, no FFN --
so each patch keeps its own content, then apply CLIP's final LN and projection.
"""
import numpy as np
import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPTokenizer

MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073])[:, None, None]
STD = torch.tensor([0.26862954, 0.26130258, 0.27577711])[:, None, None]
NAME = "openai/clip-vit-base-patch16"


class DenseCLIP:
    def __init__(self, device="cpu", res=448, name=NAME, half=None):
        self.device, self.res = device, res
        self.half = (device != "cpu") if half is None else half
        self.model = CLIPModel.from_pretrained(name).eval().to(device)
        if self.half:
            self.model.half()
        self.tok = CLIPTokenizer.from_pretrained(name)
        self.grid = res // self.model.config.vision_config.patch_size

    def preprocess(self, pil_images):
        xs = [torch.from_numpy(np.asarray(im.convert("RGB").resize((self.res, self.res)), dtype=np.float32))
              .permute(2, 0, 1) / 255.0 for im in pil_images]
        x = (torch.stack(xs) - MEAN) / STD
        return x.to(self.device, torch.float16 if self.half else torch.float32)

    @torch.no_grad()
    def text(self, prompts, batch=256):
        out = []
        for i in range(0, len(prompts), batch):
            t = self.tok(prompts[i:i + batch], padding=True, truncation=True, return_tensors="pt").to(self.device)
            # text_model + projection instead of get_text_features(): the latter changed its
            # return type across transformers versions (tensor vs. ModelOutput)
            feats = self.model.text_projection(self.model.text_model(**t).pooler_output)
            out.append(F.normalize(feats.float(), dim=-1).cpu())
        return torch.cat(out)                                           # (N, 512)

    @torch.no_grad()
    def patches(self, pil_images, value_only=True):
        vm = self.model.vision_model
        out = vm(pixel_values=self.preprocess(pil_images), output_hidden_states=True,
                 interpolate_pos_encoding=True)
        if value_only:
            last = vm.encoder.layers[-1]
            h = last.layer_norm1(out.hidden_states[-2])
            tok = last.self_attn.out_proj(last.self_attn.v_proj(h))[:, 1:]
        else:
            tok = out.last_hidden_state[:, 1:]
        tok = self.model.visual_projection(vm.post_layernorm(tok))
        tok = F.normalize(tok.float(), dim=-1)                          # (B, G*G, 512)
        return tok.transpose(1, 2).reshape(len(pil_images), -1, self.grid, self.grid).cpu()
