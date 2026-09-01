"""
H2 offline pipeline: EnCodec 24kHz encode/decode at multiple bitrates.
Outputs per input file, per bitrate:
  - reconstructed .wav  -> assets/audio/
  - token codes JSON    -> assets/tokens/
  - mel spectrogram PNG -> assets/spectrograms/
Run: python encode.py <input.wav> [input2.wav ...]
"""
import sys, os, json
import torch
import numpy as np
import soundfile as sf
import librosa, librosa.display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from encodec import EncodecModel
from encodec.utils import convert_audio

BITRATES = [1.5, 3.0, 6.0, 12.0, 24.0]   # kbps -> n_q = 2,4,8,16,32
OUT = {"audio": "assets/audio", "tokens": "assets/tokens", "spec": "assets/spectrograms"}
for d in OUT.values():
    os.makedirs(d, exist_ok=True)

model = EncodecModel.encodec_model_24khz()

def mel_png(wav_np, sr, path, title):
    S = librosa.feature.melspectrogram(y=wav_np, sr=sr, n_mels=128, fmax=sr // 2)
    S_db = librosa.power_to_db(S, ref=np.max)
    plt.figure(figsize=(6, 3))
    librosa.display.specshow(S_db, sr=sr, x_axis="time", y_axis="mel", fmax=sr // 2)
    plt.title(title); plt.tight_layout()
    plt.savefig(path, dpi=110); plt.close()

def process(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    data, sr_in = sf.read(path, dtype="float32", always_2d=True)  # [T, C]
    wav = torch.from_numpy(data.T)                                # [C, T]
    wav = convert_audio(wav, sr_in, model.sample_rate, model.channels)  # -> 24kHz mono
    sr = model.sample_rate

    # original reference outputs
    orig_np = wav.squeeze(0).numpy()
    sf.write(f"{OUT['audio']}/{stem}_original.wav", orig_np, sr)
    mel_png(orig_np, sr, f"{OUT['spec']}/{stem}_original.png", f"{stem} original")

    wav_b = wav.unsqueeze(0)  # [B,C,T]
    prev_recon = None
    for br in BITRATES:
        model.set_target_bandwidth(br)
        with torch.no_grad():
            frames = model.encode(wav_b)
            codes = torch.cat([f[0] for f in frames], dim=-1)  # [B, n_q, T_frames]
            recon = model.decode(frames).squeeze(0)            # [C, T]
        tag = f"{stem}_{br}kbps".replace(".", "_")
        recon_np = recon.squeeze(0).cpu().numpy()
        sf.write(f"{OUT['audio']}/{tag}.wav", recon_np, sr)

        # residual: what the codec threw away (time-aligned subtraction)
        L = min(len(orig_np), len(recon_np))
        resid = orig_np[:L] - recon_np[:L]
        sf.write(f"{OUT['audio']}/{tag}_residual.wav", resid, sr)

        # contribution: what the layers added at this step vs the previous bitrate
        # (RVQ is nested, so recon(b) - recon(b_prev) = sound of the added codebooks;
        #  for the first bitrate the contribution IS the foundation reconstruction)
        contrib = recon_np if prev_recon is None else recon_np[:min(len(recon_np), len(prev_recon))] - prev_recon[:min(len(recon_np), len(prev_recon))]
        sf.write(f"{OUT['audio']}/{tag}_contrib.wav", contrib, sr)
        prev_recon = recon_np
        mel_png(recon_np, sr, f"{OUT['spec']}/{tag}.png", f"{stem} {br} kbps")

        c = codes.squeeze(0).cpu().numpy()  # [n_q, T_frames]
        json.dump(
            {"file": stem, "kbps": br, "n_q": int(c.shape[0]),
             "n_frames": int(c.shape[1]), "codes": c.astype(int).tolist()},
            open(f"{OUT['tokens']}/{tag}.json", "w"))
        print(f"  {br:>5} kbps  n_q={c.shape[0]:>2}  frames={c.shape[1]}")

if __name__ == "__main__":
    files = sys.argv[1:] or ["sample.wav"]
    stems = []
    for f in files:
        print(f"processing {f}")
        process(f)
        stems.append(os.path.splitext(os.path.basename(f))[0])
    json.dump({"samples": stems, "bitrates": BITRATES},
              open("assets/manifest.json", "w"))
    print("done ->", OUT, "+ assets/manifest.json")
