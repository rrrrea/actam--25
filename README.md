# residuals

**Field recordings through a neural audio codec — residual vector quantization as compositional material.**

Live demo: **https://\<username\>.github.io/\<repo\>/** <!-- replace after Pages deploy -->

![screenshot](readme_images/screenshot.png) <!-- add a screenshot of the running app -->

Developed for the **Advanced Coding Tools and Methodologies** course (ACTAM 2025/26, Politecnico di Milano). Tested on Chrome; use headphones — most of what this app demonstrates lives in the details.

## Summary
* [Concept](#concept)
* [Why EnCodec?](#why-encodec)
* [How residual vector quantization works](#how-residual-vector-quantization-works)
* [Using the app](#using-the-app)
* [Sample curation](#sample-curation)
* [Reading the artifacts](#reading-the-artifacts)
* [Architecture](#architecture)
* [Run locally](#run-locally)
* [Add your own recordings](#add-your-own-recordings)

## Concept

Neural audio codecs are trained to compress *human* signals — speech and music — as transparently as possible. This project asks what happens when a compression system built for human sounds encounters non-human ones: birdsong, environmental textures, resonant percussion.

At high bitrates the reconstruction is near-transparent. As the bitrate drops, the codec must describe the signal with fewer discrete symbols, and its failures become audible: smeared attacks, wobbling pitches, phantom harmonics. **residuals** makes this degradation inspectable in three ways — you hear the reconstruction, you hear what was *thrown away*, and you see the entire discrete representation the sound was decoded from.

The name is a double reading. In residual vector quantization, the *residual* is the error each codebook layer passes to the next. In the glitch and microsound tradition (Kim Cascone's "aesthetics of failure"), the *residue* of a technical system is musical substance. Here the two meanings collapse into one button: the low-bitrate residual files exported by this project are the source material for an electroacoustic composition.

There is a third reading, looking forward rather than back: EnCodec tokens are the vocabulary of audio language models — MusicGen and VALL-E generate exactly these integers. The artifacts you hear at low bitrates are the resolution limit at which such models perceive sound.

## Why EnCodec?

[EnCodec](https://github.com/facebookresearch/encodec) (Meta, 2022) is a neural audio codec with an Encoder → RVQ → Decoder architecture. Two reasons it anchors this project:

1. **Its artifacts are musically interesting.** Compared to more recent codecs, EnCodec degrades earlier and more dramatically at low bitrates — which is exactly the material this project is after.
2. **It is a tokenizer, not just a codec.** Its discrete codes are what audio language models read and write; exploring them is exploring how those models hear.

The 24 kHz mono variant is used at five bandwidth settings: 1.5, 3, 6, 12, 24 kbps.

## How residual vector quantization works

The encoder maps 24,000 samples per second down to **75 latent frames per second**. Each frame is a continuous vector that must become discrete. RVQ does this in layers:

1. Codebook 1 (1024 entries) finds the nearest entry to the frame vector and records its **index** — one integer, the first token.
2. Quantization leaves an error (the *residual*). Codebook 2 quantizes that residual; codebook 3 quantizes what's left, and so on.
3. The bitrate decides how many layers run: **1.5 kbps → 2 codebooks, 24 kbps → 32 codebooks.**

A 10-second recording at 1.5 kbps is therefore an integer matrix of shape `[2 × 750]` — 1,500 table lookups, and *nothing else*. The decoder reconstructs the waveform from those indices alone. Everything that did not fit into the budget is what you hear as artifact.

Because the quantizer is **nested** — the first *k* codebooks of a 24 kbps encoding *are* the k-layer encoding — the bitrate ladder in this app is literally a tour up the RVQ stack.

## Using the app

- **play** starts all variants of the current recording simultaneously; the bitrate keys just re-route which one you hear, so every comparison is perfectly time-aligned and click-free.
- **spacebar** — instant A/B between the original and the current bitrate. The fastest way to hear a difference.
- **residual** — hear `original − reconstruction`: the part of the signal the codec threw away. Loud and textured at 1.5 kbps, near silence at 24.
- **layers** — hear what each step up the ladder *adds*: 1.5k plays the two-codebook foundation, 3k plays only what layers 3–4 contribute, and so on up to the airy final corrections of layers 17–32. Pressed in order, this is the RVQ algorithm as a listening experience.
- **token grid** — every colored cell is one integer. Hover to read its layer, frame, time and value; the red playhead shows which column you are hearing.

## Sample curation

The four recordings are not interchangeable — each probes a different failure mode of the codec:

| recording | signal type | what it tests | expected result |
|---|---|---|---|
| `attention-male` | speech | the codec's home turf (control) | stable at every bitrate — it was built for this |
| `percussion-underworld` | dense transients + resonance | temporal resolution (75 frames/s ≈ 13 ms per frame) | smeared attacks, pre-echo at low bitrates |
| `bird_whistle` | high, clean pitch sweeps | frequency precision — quantization error has nowhere to hide on a pure tone | pitch wobble, phantom partials |
| `morning` | broadband environmental texture | the masking floor (control) | smallest audible change — noise hides quantization noise |

Read together: degradation speed is not uniform, and its ordering mirrors the codec's training distribution. Speech survives best; transients and pure tones break first; texture gets away with it. That ordering *is* the project's concept, made audible.

## Reading the artifacts

A short listener's dictionary, artifact → mechanism:

- **Smeared or softened attacks** (percussion, low bitrates) — one latent frame spans ~13 ms; a drum hit is shorter than the codec's temporal pixel.
- **Pitch wobble and phantom partials** (bird whistle) — with few codebooks, the nearest available code jumps between frames; a clean sweep becomes a staircase with ghosts.
- **Metallic, granular sheen** (everywhere at 1.5 kbps) — two coarse codebooks per frame force the decoder to hallucinate texture from a tiny vocabulary.
- **Residual that sounds like a whispering double of the original** — what you are hearing is precisely the information that did not fit; as bitrate rises it thins into hiss, then silence.

## Architecture

Fully static — no backend, no runtime inference. Heavy computation happens once, offline:

```
scripts/encode.py  (Python, offline)          browser (static)
─────────────────────────────────    ────────────────────────────────
wav → EnCodec encode/decode      →   assets/audio/*.wav            → Web Audio, gain-switched A/B
    → original − reconstruction  →   assets/audio/*_residual.wav
    → recon(b) − recon(b_prev)   →   assets/audio/*_contrib.wav    → "layers" mode
    → codes tensor dump          →   assets/tokens/*.json          → canvas heatmap + hover
    → mel spectrograms           →   assets/spectrograms/*.png
    → manifest.json              →   drives the sample list
```

## Run locally

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch encodec librosa soundfile matplotlib
python3 -m http.server 8000        # then open http://localhost:8000
```

## Add your own recordings

```bash
source .venv/bin/activate
python scripts/prep.py your_recording.wav      # trim silence, normalize
python scripts/encode.py sound_source_24khz/*.wav
```

Keep clips ≤ 30 s (CPU inference over five bitrates). `encode.py` regenerates `assets/manifest.json` from the files passed on the command line, so list **every** sample you want visible.

---

*ACTAM 2025/26 · Music and Acoustic Engineering, Politecnico di Milano*
