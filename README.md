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

### `Git` clone

Clone this repo through github:
```bash
git clone https://github.com/HeejongBong/ladyns.git
```

### `Python` install

Install package `ladyns` using setup.py script:
```bash
python setup.py install
```

## Experimental data 

The data are available in `/data/`. The data file consists of `lfp_bred_1.mat`, `lfp_bred_2.mat`, and `lfp_bred_3.mat` which are the beta band-passed filtered LFP in PFC and V4 for 3 thousand trials, respectively. These data are the results of the preprocess by `/example/4_0_preprocess_experimental_data.ipynb` from the original data collected by Khanna, Scott, and Smith (2020) [[2](#KSS19)].

## Reproducible Ipython notebooks

The scripts are available in `/example/`. The scripts for the simulation analysis are provided in `Python` notebook from `3_1_...ipynb` to `3_3_...ipynb`. The scripts for the experimental data analysis are provided in `Python` notebook `4_1_analyze_experimental_data.ipynb`

## References

<a name="BYSVK20"> [1] Bong, H., Yttri, E., Smith, M. A., Ventura, V., & Kass, R. E. (2020). Latent Dynamic Factor Analysis of High-Dimensional Neural Recordings. *Submitted to Annals of Applied Statistics*. </a>

<a name="KSS19"> [2] Khanna, S. B., Scott, J. A., & Smith, M. A. (2020). Dynamic shifts of visual and saccade signals in prefrontal cortical regions 8Ar and FEF. *Journal of neurophysiology*. 124.6 (2020): 1774-1791. </a>
