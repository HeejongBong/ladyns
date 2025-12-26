# LaDynS: Latent Dynamic Analysis via Sparse Banded Graphs

This repo consists of 
1. Python package `ladyns` which implements LaDynS [[1](#BYSVK21)]
2. Experimental data analyzed in [[1](#BYSVK21)]
3. Reproducible IPython notebooks for simulation and experimental data analysis in [[1](#BYSVK21)]

## Install

### Prerequisite

Package `ladyns` requires:
1. `Python` >= 3.5 
2. `numpy` >= 1.8
3. `matplotlib` and `scipy`.

While `ladyns` can run under the most recent dependencies, the current example codes require `numpy<2` and `scipy==1.11.1`.

### `Git` clone

Clone this repo through github:
```bash
git clone https://github.com/HeejongBong/ladyns.git
```

### `conda` environment setup

```bash
conda create -n ladyns python=3.11 -c conda-forge
conda activate ladyns
conda install -c conda-forge 'scipy==1.11.1' 'numpy<2' matplotlib openblas libblas liblapack pkg-config cmake compilers meson-python meson ninja pandas pyarrow tqdm scikit-learn ffmpeg h5py
```

### `Python` install

Install package `ladyns` using `pip`:
```bash
cd directory/to/ladyns
pip install -v .
```

## Experimental data 

The data are available in `/data/`. The data file consists of `lfp_bred_1.mat`, `lfp_bred_2.mat`, and `lfp_bred_3.mat` which are the beta band-passed filtered LFP in PFC and V4 for 3 thousand trials, respectively. These data are the results of the preprocess by `/example/4_0_preprocess_experimental_data.ipynb` from the original data collected by Khanna, Scott, and Smith (2020) [[2](#KSS19)].

## Reproducible Ipython notebooks

The scripts are available in `/example/`. The scripts for the simulation analysis are provided in `Python` notebook from `1_1_...ipynb` to `3_8_...ipynb`. The scripts for the experimental data analysis are provided in `Python` notebook from `4_1_...ipynb` to `4_2_...ipynb`.

## References

<a name="BYSVK20"> [1] Bong, H., Yttri, E., Smith, M. A., Ventura, V., & Kass, R. E. (2025+). Cross-Population Amplitude Coupling in High-Dimensional Oscillatory Neural Time Series. *Submitted*. </a>

<a name="KSS19"> [2] Khanna, S. B., Scott, J. A., & Smith, M. A. (2020). Dynamic shifts of visual and saccade signals in prefrontal cortical regions 8Ar and FEF. *Journal of neurophysiology*. 124.6 (2020): 1774-1791. </a>
