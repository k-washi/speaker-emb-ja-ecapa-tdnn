"""torch.hub configuration."""

dependencies = ["torch"]

import torch
from pathlib import Path
import gdown

from speakeremb_ja_ecapatdnn.ecapatdnn import _SpeakerEmbeddingJa


def ecapatdnn_ja_l512(progress: bool = True, pretrained: bool = True) -> _SpeakerEmbeddingJa:
    model = _SpeakerEmbeddingJa(hidden_size=512)
    if pretrained:
        output_dir = Path("/tmp/speaker-emb-ja-ecapa-tdnn")
        output_dir.mkdir(exist_ok=True, parents=True)
        output_fp = output_dir / "ecapatdnn_ja_l512.pth"
        gdown.download(
            "https://drive.google.com/u/1/uc?id=1h5cKOZyqXWRz203IeJysuJQVrVPfueZw",
            str(output_fp),
            quiet=False
        )
        model.model.load_state_dict(torch.load(output_fp))
    
    model.model.eval()
    model.model.preprocess.eval()
    for param in model.model.parameters():
        param.requires_grad = False
    return model
    