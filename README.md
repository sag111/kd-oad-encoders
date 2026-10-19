# KD-OAD Encoders

Compact RGB-only encoders for online action detection, distilled from the
original TeSTra appearance and motion feature extractors.

- [Pretrained checkpoints and model card on Hugging Face](https://huggingface.co/elnmax/kd-oad-encoders)
- Paper: forthcoming

The student encoder is based on X3D XS and accepts RGB frames only. TeSTra is
used as the online detection head. The repository includes preprocessing,
feature-export, target-generation, and evaluation scripts for THUMOS14 and
TVSeries.

## Installation

Requirements:

- Python 3.10 or newer
- `ffmpeg`
- a CUDA-compatible GPU for the examples below

Clone the repository, then initialize only the TeSTra submodule:

```bash
git clone https://github.com/sag111/kd-oad-encoders.git
cd kd-oad-encoders
git submodule update --init testra
```

If the repository has already been cloned, initialize TeSTra with:

```bash
git submodule update --init testra
```

Do not add `--recursive`: TeSTra declares its own optional submodule, which is
not required for this project.

Create the project virtual environment and install both packages:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e kdoad
pip install -e testra
```

## Checkpoints

Download the pretrained X3D encoders and TeSTra heads from
[Hugging Face](https://huggingface.co/elnmax/kd-oad-encoders):

```bash
hf download elnmax/kd-oad-encoders \
  --local-dir checkpoints \
  --include "*.pth" "*.pt"
```

The command downloads the following files:

```text
checkpoints/
├── thumos_x3d.pt
├── thumos_testra.pth
├── tvseries_x3d.pt
└── tvseries_testra.pth
```

## Evaluation

Run all commands from the repository root with the virtual environment
activated. The examples use GPU `0`, export 2048-dimensional X3D features at
4 FPS, and save them under each dataset's `features/repr_x3d_test` directory.

Dataset videos are not included. Obtain them from the official
[THUMOS14](https://www.crcv.ucf.edu/THUMOS14/) and
[TVSeries](https://homes.esat.kuleuven.be/psi-archive/rdegeest/TVSeries.html)
pages.

### THUMOS14

1. Place the source videos in:

   ```text
   datasets/THUMOS14/raw/validation/
   datasets/THUMOS14/raw/test/
   datasets/THUMOS14/raw/videos/
   ```

2. Convert the videos to 224x224 MP4 files at 24 FPS:

   ```bash
   cd datasets/THUMOS14
   bash preprocess.sh
   cd ../..
   ```

3. Export X3D features:

   ```bash
   OUTPUT_FOLDER=features/repr_x3d_test \
   DEVICE=0 \
   ./kdoad/scripts/export_features_thumos.sh
   ```

4. Download the official per-frame TeSTra targets:

   ```bash
   python -m gdown \
     "https://drive.google.com/uc?id=1L6I1x6J5YORL9H6Gc4193GGbdBSNo34k" \
     -O datasets/THUMOS14/target_perframe.zip
   unzip datasets/THUMOS14/target_perframe.zip \
     -d datasets/THUMOS14/target_perframe
   rm datasets/THUMOS14/target_perframe.zip
   ```

5. Evaluate the TeSTra checkpoint in stream mode:

   ```bash
   DATA_ROOT=../datasets/THUMOS14 \
   GPU=0 \
   ./testra/scripts/test_checkpoint.sh \
     thumos \
     ../checkpoints/thumos_testra.pth
   ```

### TVSeries

1. Place the original MKV episodes in:

   ```text
   datasets/TVSeries/raw/
   ```

   Filenames must match the train, validation, and test episode lists in
   `datasets/TVSeries/preprocess.sh`.

2. Convert the videos to 224x224 MP4 files at 24 FPS:

   ```bash
   cd datasets/TVSeries
   bash preprocess.sh
   cd ../..
   ```

3. Export X3D features:

   ```bash
   OUTPUT_FOLDER=features/repr_x3d_test \
   DEVICE=0 \
   ./kdoad/scripts/export_features_tvseries.sh
   ```

4. Generate per-frame TeSTra targets from `datasets/TVSeries/labels.csv`:

   ```bash
   ./kdoad/scripts/generate_target_perframe.sh
   ```

5. Evaluate the TeSTra checkpoint in stream mode:

   ```bash
   DATA_ROOT=../datasets/TVSeries \
   GPU=0 \
   ./testra/scripts/test_checkpoint.sh \
     tvseries \
     ../checkpoints/tvseries_testra.pth
   ```

The evaluation script changes its working directory to `testra`; therefore,
`DATA_ROOT` and checkpoint paths in these commands are relative to that
directory.

## Results

| Dataset | Metric | Best run | 10 runs |
| --- | --- | ---: | ---: |
| THUMOS14 | mAP | 66.3% | 65.9 ± 0.2% |
| TVSeries | mcAP | 86.2% | 86.0 ± 0.1% |

## Citation

Citation details will be added after the paper is published.

## License

The original code and weights are released under the [MIT License](LICENSE).
The [TeSTra](testra) submodule is licensed separately under the
[Apache License 2.0](testra/LICENSE).
