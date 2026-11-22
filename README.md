# VAE for Bearing Vibration

A 1D-convolutional variational autoencoder (VAE) trained only on **healthy** rolling bearings from the Paderborn University bearing dataset. It is used to:

1. generate new healthy vibration spectra, and
2. detect damaged bearings by their reconstruction error (anomaly detection).

Everything is in [`generative_model.ipynb`](generative_model.ipynb). The figures it saves are in [`figures/`](figures/).

## Dataset download (Zenodo)

The notebook uses the Zenodo copy of the Paderborn bearing data: **https://zenodo.org/records/15845309** (DOI [10.5281/zenodo.15845309](https://doi.org/10.5281/zenodo.15845309), CC BY 4.0). It contains one `.rar` archive per bearing (31 archives, about 5.2 GB in total). Only the 12 bearings listed below are needed, about 2 GB.

1. **Download the 12 archives**, either by hand from the record's *Files* section or from the terminal:

   ```bash
   mkdir -p downloads && cd downloads
   for b in K001 K002 K003 K004 K005 K006 KA04 KA15 KA16 KI04 KI14 KI16; do
     curl -L -o "$b.rar" "https://zenodo.org/records/15845309/files/$b.rar?download=1"
   done
   ```

   Zenodo lists an MD5 checksum for every file. You can compare it with `md5 K001.rar` (macOS) or `md5sum K001.rar` (Linux).

2. **Extract them into `data/paderborn/`.** You need a RAR tool, for example `unar` (`brew install unar`) or 7-Zip (`7z`):

   ```bash
   mkdir -p ../data/paderborn
   for f in *.rar; do unar -f -o ../data/paderborn "$f"; done     # or: 7z x "$f" -o../data/paderborn
   ```

3. **Check the layout.** Each bearing must have its own folder:

   ```
   data/paderborn/
   ├── K001/
   │   ├── N15_M07_F10_K001_1.mat
   │   ├── ...
   │   └── N15_M07_F10_K001_20.mat
   ├── K002/
   └── ...
   ```

   The notebook only reads the 20 recordings per bearing taken at operating condition **N15_M07_F10**: 1500 rpm, 0.7 Nm load torque, 1000 N radial force. Other files in the folders are ignored. From each `.mat` file it uses only the `vibration_1` channel, sampled at 64 kHz for 4 s.

`data/paderborn/` is in `.gitignore`, so the raw data is never committed.

## Bearings used

| Bearing | Condition | Role in this project |
|---|---|---|
| K001, K002, K003, K004 | Healthy | **Training** (recordings 1–16) and validation (recordings 17–20) |
| K005, K006 | Healthy | Test: unseen healthy bearings |
| KA04, KA15, KA16 | Outer-race damage | Test: damaged |
| KI04, KI14, KI16 | Inner-race damage | Test: damaged |

The damaged bearings all have real damage from accelerated lifetime tests, not damage made artificially. The split is by **bearing**, so no test bearing is ever seen during training.

## The 10 kHz decision

Each 4-second recording is cut into 64 ms windows (4096 samples). Each window is turned into a log power spectrum, and **only 0–10 kHz is kept**. The 640 FFT bins up to 10 kHz are averaged in groups of 5, giving 128 bins of 78 Hz each, which is the VAE input.

Why the cut at 10 kHz:

- **Above about 10–11 kHz the spectrum is a flat noise floor** (Figure 1). Its level depends on the *bearing / recording session*, not on the damage. Among the healthy bearings alone, the 15–30 kHz floor ranges from −5.40 (K001) to −6.60 (K004) in log10 power, a 16× difference. Within a bearing it hardly changes (standard deviation 0.01–0.02 across its 20 recordings).
- **Most damaged bearings happen to have a low floor** (−6.5 to −6.9), while the training bearings K001 and K003 have a high one (about −5.4). If the high band were kept, the VAE could "detect damage" just by recognising the floor level. That would be a recording artefact, not a real fault signature.
- **The structural resonances and the fault-related differences sit below 10 kHz** (peaks near 1, 3.3, 6, 7.8 and 10 kHz).

The trade-off is that any fault information above 10 kHz is thrown away. Even inside 0–10 kHz, overall loudness still varies by bearing, and the notebook shows the VAE partly picks up on it.

## How to run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter lab generative_model.ipynb      # or open it in VS Code
```

Tested with Python 3.15, torch 2.15, numpy 2.5, scipy 1.18, pandas 3.1, scikit-learn 1.9, matplotlib 3.11 and seaborn 0.13. The notebook uses CUDA or Apple MPS if available and falls back to the CPU. Training takes 60 epochs on about 4,000 spectra and runs in a few minutes on a laptop.

## Results in short

- Generated spectra match the mean healthy spectrum closely but are **over-smoothed**: their spread is only about two-thirds of the real spread.
- Reconstruction error separates every damaged bearing from unseen healthy K005/K006 (**AUROC 0.92–1.00**).
- **Prediction (written before training):** KA15 and KI14 would be the hardest damaged bearings to detect. They came out second and third hardest, but the prediction missed the hardest, KI04 (AUROC 0.916). A loudness measure chosen *after* seeing the results explains all three. That makes it a hypothesis to test on new bearings, not a confirmation.
- **Failure mode:** healthy K005 scores about 50% higher than the training bearings and overlaps the quietest damaged bearings, which would cause false alarms.

See the interpretation cells and the summary at the end of the notebook for details.

## Citation

Please cite **both** the Zenodo copy used here and the original dataset paper:

- **Zenodo copy (data source for this project):**
  van den Hoogen, J. (2025). *Paderborn Bearing Dataset* (Version v1) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.15845309. Licence: CC BY 4.0.

- **Original dataset (Paderborn University):**
  Lessmeier, C., Kimotho, J. K., Zimmer, D., & Sextro, W. (2016). Condition monitoring of bearing damage in electromechanical drive systems by using motor current signals of electric motors: A benchmark data set for data-driven classification. *Proceedings of the European Conference of the Prognostics and Health Management Society*, 3(1).

The data was recorded by Lessmeier et al. at the Chair of Design and Drive Technology (KAt), Paderborn University. The Zenodo record is a re-upload of that data by Jurgen van den Hoogen (Osnabrück University).
