# pyHREBSD
HREBSD implementation in python mirrored after EMSoft in the SEM/EMHREBSD.f90 file.

### Conda Env (windows, cuda version = 12.4)
```
conda create -n hrebsd python=3.12 numpy matplotlib tqdm scipy scikit-image -c conda-forge
conda activate hrebsd
pip install mpire[dill]
conda install pytorch torchvision torchaudio kornia pytorch-cuda=12.4 -c pytorch -c nvidia
```
### Conda Env (Mac or no CUDA)
```
conda create -n hrebsd python=3.12 numpy matplotlib tqdm scipy scikit-image pytorch kornia -c pytorch -c conda-forge
conda activate hrebsd
pip install mpire[dill]
```

## TODO

- Preprocessing
  - Write windowing function for the ROIs
  - Write the bandpass filters for the ROIs
- Core functionality
  - Write the pattern center refinement function
- Verifications
  - Objective function, constrain function, and minimization function need to be checked
  - Cross-correlation function needs to be checked
  - Peak interpolation needs to be checked
