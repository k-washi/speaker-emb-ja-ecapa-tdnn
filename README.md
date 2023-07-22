# speaker-emb-ja-ecapa-tdnn

日本語話者で学習した話者Embedding

学習リポジトリ: [spk_emb_ja](https://github.com/k-washi/spk_emb_ja)

# 推論の実行

```python
import torch
import torch.nn.functional as F
import torchaudio
from ecapatdnn import SpeakerEmbeddingJa

audio_path = "sample.wav"
ckpt_path = "ecapatdnn_l512_n4690_st.pth"
hidden_size = 512

model = SpeakerEmbeddingJa(ckpt_path, hidden_size)

wave, sr = torchaudio.load(audio_path)
wave = torchaudio.transforms.Resample(sr, model.sample_rate)(wave) # (batch:1, wave length)


emb = model.extract_embedding(wave) # (batch:, hidden_size)
emb = F.normalize(torch.FloatTensor(emb), p=2, dim=1).detach().cpu()

# embedding similarity
score = torch.mean(torch.matmul(emb, emb.T)) # 1
```

# モデル

|モデル|speaker num|hidden_size|-|
|-|-|-|-|
|ecapatdnn_l512_n4690_st.pth|4960|512||
|ecapatdnn_l128_n2340_clean_st.pth|2340|128||
|ecapa_tdnn_l512_n4690_st_volume_aug.pth|4960|512|音量のデータ拡張追加|


```
pip install gdown

# ecapatdnn_l512_n4690_st.pth
gdown https://drive.google.com/u/1/uc?id=1h5cKOZyqXWRz203IeJysuJQVrVPfueZw -O ecapatdnn_l512_n4690_st.pth

# ecapatdnn_l128_n2340_clean_st.pth
gdown https://drive.google.com/u/1/uc?id=1Qa0lqrKduUCJzagqe59fQ5R8-xQmeIVG -O ecapatdnn_l128_n2340_clean_st.pth

# ecapa_tdnn_l512_n4690_st_volume_aug.pth
gdown https://drive.google.com/u/1/uc?id=1QrwdyDRlkFHqjKBeZ5HbreaOI_thrbTv -O ecapa_tdnn_l512_n4690_st_volume_aug.pth
```

# ライセンス

使用する際は、このリポジトリーにいいねして、ライセンス明記してください。

```
[speaker-emb-ja-ecapa-tdnn](https://github.com/k-washi/speaker-emb-ja-ecapa-tdnn)
```

と書くだけでOKです。

FineTuningなどの事前学習モデルとして使用する場合も、同様です。

これらを利用したとしても、コードの公開義務はありません！