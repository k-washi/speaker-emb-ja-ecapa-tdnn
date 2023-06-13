# speaker-emb-ja-ecapa-tdnn

日本語話者で学習した話者Embedding

学習リポジトリ: [spk_emb_ja](https://github.com/k-washi/spk_emb_ja)

# 推論の実行

```python
import torchaudio
from ecapatdnn import SpeakerEmbedding

audio_path = "sample.wav"
sample_rate = 16000
ckpt_path = "ecapa_tdnn.pth"

wave, sr = torchaudio.load(audio_path)
wave = torchaudio.transforms.Resample(sr, sample_rate)(wave).unsqueeze(0)

model = SpeakerEmbedding(ckpt_path)
emb = model.vectorize(wave)
```