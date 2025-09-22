# Scaling with Multiple Scaling Factors and Dynamic Time Warping in Time Series Searching

![PSDTW intution](figures/psdtw-intuition.pptx.svg)
- The above figure shows the intuition of the necessity of our novel distance measure PSDTW.

![PSDTW example](figures/psdtw-ex.png)
- The red time series is the original time series. We introduce piecewise scaling (ps) to it and derive the blue time series.
  - Since PSDTW tries to divide two time series into continuous pairs of segments, we can know the cut points and the corresponding scaling factor.
    -  The scaling factor can be calculated by the lengths of the segment in the same aligned pair.
      
# Notifications
Dates on [AoE](https://www.timeanddate.com/time/zones/aoe) Time Zone
- 2025-08-29: Submitted to [IEEE BigData 2025](https://conferences.cis.um.edu.mo/ieeebigdata2025/).

## Pending Tasks
- [] 2025-08-29: Make video presentation.
- [] 2025-08-29: Make slides.

# Install
```
conda create -n ksfdtw python=3.12
conda activate ksfdtw
conda install -c conda-forge dtaidistance
pip install pandas
conda install -c conda-forge tslearn
pip install tqdm # For progress bar
pip install pyts
conda install -c conda-forge sktime  
conda install -c conda-forge aeon
conda install h5py
# Export your active environment to a new file:
conda env export > environment.yml
# Deactivate the environment
conda deactivate
# Delete the environment if needed.
conda env remove -n ksfdtw

# Or readers can make a new environment from our environment. 
# But different platforms may have their own platform-specific packages that may cause error if importing `environments.yml` directly.
conda env create --name envname --file=environments.yml
```


# Project Structure
- Explore our PSDTW: [main.ipynb](https://github.com/cyuab/k-scaling-factor-dtw/blob/main/code/main.ipynb)
- Data Visualization: [class_representative.ipynb](https://github.com/cyuab/k-scaling-factor-dtw/blob/main/code/class_representative_fig.ipynb)
- Data Processing: [data_processing.ipynb](https://github.com/cyuab/k-scaling-factor-dtw/blob/main/code/data_processing.ipynb)
- Query Experiment: [querying.ipynb](https://github.com/cyuab/k-scaling-factor-dtw/blob/main/code/querying.ipynb) (P.S., A simple query: [querying_ex.ipynb](https://github.com/cyuab/k-scaling-factor-dtw/blob/main/code/querying_ex.ipynb))
- Important folders and files in this repository are listed as belows:
  ```bash
  .
  ├── code
  │   ├── class_representative_fig.ipynb
  │   ├── data_processing.ipynb
  │   ├── dtw_bands_fig.ipynb
  │   ├── ksfdtw
  │   │   ├── distance_measures.py
  │   │   ├── lower_bounds.py
  │   │   └── utils.py
  │   ├── lb_keogh_vs_lb_shen_fig.ipynb
  │   ├── main.ipynb
  │   ├── querying_ex.ipynb
  │   ├── querying.ipynb
  ├── data
  ├── data_intermediate
  ├── environment.yaml
  ├── figures
  ├── README.md
  └── results
  ```

# Corresponding Paper
-It will be updated after paper acceptance.

## Figures/Tables in the Paper
- Figures 1, 4: [main.ipynb](https://github.com/cyuab/k-scaling-factor-dtw/blob/main/code/main.ipynb)
- Figure 3: [dtw_bands_fig.ipynb](https://github.com/cyuab/k-scaling-factor-dtw/blob/main/code/dtw_bands_fig.ipynb)
- Figure 5: [lb_keogh_vs_lb_shen_fig.ipynb](https://github.com/cyuab/k-scaling-factor-dtw/blob/main/code/lb_keogh_vs_lb_shen_fig.ipynb)
- Figure 6: [class_representative_fig.ipynb](https://github.com/cyuab/k-scaling-factor-dtw/blob/main/code/class_representative_fig.ipynb)
- Figure 7: [data_processing.ipynb](https://github.com/cyuab/k-scaling-factor-dtw/blob/main/code/data_processing.ipynb)


# Resources
1. [aeon](https://www.aeon-toolkit.org/en/stable/index.html)
    - [Distances - aeon 1.2.0 documentation](https://www.aeon-toolkit.org/en/stable/api_reference/distances.html)
    - [dtw_distance - aeon 1.2.0 documentation](https://www.aeon-toolkit.org/en/stable/api_reference/auto_generated/aeon.distances.dtw_distance.html)
1. [DTAIDistance](https://dtaidistance.readthedocs.io/en/latest/index.html)
    - [Dynamic Time Warping (DTW) — DTAIDistance 2.3.9 documentation](https://dtaidistance.readthedocs.io/en/latest/usage/dtw.html)
1. [tslearn](https://tslearn.readthedocs.io/en/stable/index.html)
    - [tslearn.metrics — tslearn 0.6.4 documentation](https://tslearn.readthedocs.io/en/stable/gen_modules/tslearn.metrics.html#module-tslearn.metrics)
    - [Dynamic Time Warping — tslearn 0.6.4 documentation](https://tslearn.readthedocs.io/en/stable/user_guide/dtw.html)
1. [pyts](https://pyts.readthedocs.io/en/stable/index.html)
    - [pyts.metrics.dtw — pyts 0.13.0 documentation](https://pyts.readthedocs.io/en/stable/generated/pyts.metrics.dtw.html)
1. [sktime](https://www.sktime.net/en/stable/)
    - [dtw_distance — sktime documentation](https://www.sktime.net/en/stable/api_reference/auto_generated/sktime.distances.dtw_distance.html)
1. FastDTW
    - [Original Java implementation](https://github.com/rmaestre/FastDTW)
    - [Python implementation](https://github.com/slaypni/fastdtw)
1. [dtw-python · PyPI](https://pypi.org/project/dtw-python/)
1. [dtw · PyPI](https://pypi.org/project/dtw/)
    
# Contacts
- It will be updated after paper acceptance.

test


# tmux
```
# https://www.ruanyifeng.com/blog/2019/10/tmux.html
# https://superuser.com/questions/249659/how-to-detach-a-tmux-session-that-itself-already-in-a-tmux
# https://www.redhat.com/en/blog/introduction-tmux-linux

$ tmux ls
$ tmux new -s <session-name>
# Detach
Ctrl+B then D
$ tmux attach -t <session-name>
$ tmux kill-session -t <session-name>

```


