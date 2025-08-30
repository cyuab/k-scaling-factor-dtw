# Scaling with Multiple Scaling Factors and Dynamic Time Warping in Time Series Searching

![PSDTW intution](figures/psdtw-intuition.pptx.svg)
- The above figure shows the intuition of the necessity of our novel distance measure PSDTW.
![PSDTW example](figures/psdtw-ex.png)
- The red time series is the original time series. We introduce piecewise scaling to it and derive the blue time series.
  - So, they share the same set of segments, each with a different scaling factor.
  - Our algorithm (in `../code/main.ipynb`) can discover the cutting points of segments.
    - `[((0, 21), (0, 22)), ((21, 33), (22, 29)), ((33, 50), (29, 50))]`
      -  E.g., `((0, 21), (0, 22))` indicates that the first segment in the red time series starts at 0 (inclusive) and ends at 21 (exclusive), while the first segment in the blue one starts at 0 (inclusive) and ends at 22 (exclusive).
      -  The scaling factor can be calculated by the lengths of the segment in the same aligned pair.
        - E.g., the scaling factor of the last aligned pair is $(50-33)/(50-20) \approx 0.56$. The last segment of the red has been compressed to that of the blue.     

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
- Data Visualization: class_representative.ipynb
- Data Processing: data_processing.ipynb
- Important folders and files in this repository are listed as belows:
  ```bash
  ├── code
  │   ├── data-exploration.ipynb # Explore GunPoint dataset 
  │   ├── ksfdtw.py # Custom libraries
  │   ├── main.ipynb # Explore manipulation of time series in Python
  │   ├── querying.ipynb # Experiment
  │   └── testing # The scripts under this folder is for development purpose and only for book-keeping purpose.
  ├── data # Processed dataset after processing in "data-exploration.ipynb "
  │   └── gunpoint_preprocessed.npz
  ├── environment.yaml # Store the python environment
  ├── figures # Figures for the paper and this `README.md`.
  ├── README.md # Here
  └── results # Results generated from "querying.ipynb"
  ```

# Corresponding Paper
-It will be updated after paper acceptance.

## Figures/Tables in the Paper
- Figures 1, 3, 4, 5, 6 can be found in `../main.ipynb`.
- Figures 7 (its raw figures), 8 can be found in `../data-exploration.ipynb`.

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

---
---
---








