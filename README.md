# speaker-emb-ja-ecapa-tdnn

日本語話者で学習した話者Embedding

学習リポジトリ: [spk_emb_ja](https://github.com/k-washi/spk_emb_ja)

# 推論の実行

```python
import torch
import torch.nn.functional as F
import torchaudio
from ecapatdnn import SpeakerEmbedding

audio_path = "sample.wav"
sample_rate = 16000
ckpt_path = "ecapa_tdnn.pth"

wave, sr = torchaudio.load(audio_path)
wave = torchaudio.transforms.Resample(sr, sample_rate)(wave) # (batch:1, wave length)

model = SpeakerEmbedding(ckpt_path)
emb = model.vectorize(wave) # (batch:1, 128)
emb = F.normalize(torch.FloatTensor(emb), p=2, dim=1).detach().cpu()

# embedding similarity
score = torch.mean(torch.matmul(emb, emb.T)) # 1
```