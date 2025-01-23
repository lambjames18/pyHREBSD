# pyHREBSD
HR-EBSD calculations implementing the inverse compositional Gauss-Newton optimization routine for determining the linear homography required to warp a target EBSP to match a reference EBSP in python. This code follows the HR-EBSD calculations outlined in the [ATEX](http://www.atex-software.eu) EBSD software developed by Jean-Jacques Fundenberger and Benoit Beausir. The code supports both vectorized GPU routines (through the `pytorch` package) and parallelized CPU routines (through the `mpire` package).

### Conda Env (windows, cuda version = 12.4)
```
conda create -n hrebsd python=3.12
conda activate hrebsd
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
conda install  numpy matplotlib tqdm scipy scikit-image kornia -c conda-forge
pip install mpire[dill]

```
### Conda Env (Mac or no CUDA)
```
conda create -n hrebsd python=3.12 numpy matplotlib tqdm scipy scikit-image pytorch kornia -c pytorch -c conda-forge
conda activate hrebsd
pip install mpire[dill]
```
### File descripitions
- `get_homography_cpu.py`: Contains the code for running the inverse compositional Gauss–Newton (IC-GN) algorithm for determining the homographies that warp target patterns to a reference pattern. Inside this file, the method `run_single` of the `ICGNOptimizer` class contains the actual algorithm useed to determine the homographies.
- `get_homography_gpu.py`: Same thing but for the GPU. Note that the GPU version currently does not support creating an initial guess of the homography. Inside this file, the method `run` of the `ICGNOptimizer` class contains the actual algorithm useed to determine the homographies.
- `bspline_gpu.py`: Contains all of the core GPU functions that are used during the IC-GN algorithm on the GPU.
- `warp.py`: Contains helper functions for image warping, coordinate warping, homography shape functions, and a custom Spline class. Functions for the CPU and GPU are contained here.
- `Data.py`: Contains a simgple UP2 class for reading up2 files and processing the EBSPs contained within.
- `utilities.py`: Numerous helper functions for viewing results, reading/manipulating patterns, pattern center conversions, ang/up2 reader, patttern sharpness calculation, stiffness tensor creating, etc.
- `segment.py`: Methods for segmenting grains in an EBSD dataset. This is experimental. Future use will use this function to separate an HR-EBSD calculation into individual grains.
- `rotations.py`: Conversions between different orientation representations. Copied from the pyebsdindex python package developed by Dave Rowenhorst.
- `conversions.py`: Contains conversion functions needed for the IC-GN algorithm. Such as homography to deformation gradient, deformation gradient to strain, translation and rotation to homography, etc.
- `[...]_runner.py`: A script for running the HREBSD calculation. These scripts are tailored for specific experiments/datasets, but showcase the inputs needed in order to run a scan.

All other scripts are either deprecated, for development, or for testing features. Note that this repository is actively under development and there will likely be breaking changes.

## TODO

- Fixed and dynamic projection geometries (for the pattern center shifts) are both supported, but the dynamic case has not been vetted.
- The geometry considerations for converting the strain values from the detector frame to the sample frame exist, but its possible they need to be changed.
- Strain values on the SiGe standard sample show agreement with expected values, but more verification is needed here.
