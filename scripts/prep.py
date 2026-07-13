import sys, librosa, soundfile as sf, numpy as np

for path in sys.argv[1:]:
    y, sr = librosa.load(path, sr=None, mono=True)
    yt, _ = librosa.effects.trim(y, top_db=35)      # 掐头去尾静音
    yt = yt / np.abs(yt).max() * 0.9                # 峰值归一到 0.9
    out = path.rsplit("/", 1)[-1].rsplit(".", 1)[0] + "_prep.wav"
    sf.write(out, yt, sr)
    print(f"{out}: {len(yt)/sr:.1f}s")