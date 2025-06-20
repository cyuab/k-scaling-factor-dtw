# Scaling with Multiple Scaling Factors and Dynamic Time Warping in Time Series Searching
<!-- https://stackoverflow.com/questions/39777166/display-pdf-image-in-markdown -->
<!-- for d in *.pdf ; do inkscape --without-gui --file=$d --export-plain-svg=${d%.*}.svg ; done -->
![PSDTW intution](figures/psdtw-intuition.pptx.svg)
- This figure shows intuition of the necessity of our novel distance measure PSDTW.
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
- 2025-06-07 Slides will be uploaded here later.
- 2025-06-06 Submitted to [ICDM 2025](https://www3.cs.stonybrook.edu/~icdm2025/index.html).

# Install
```
conda create -n ksfdtw python=3.12
conda activate ksfdtw
conda install -c conda-forge dtaidistance
pip install pandas
conda install -c conda-forge tslearn
pip install tqdm
pip install pyts
# Other common time series libraries that are not used in the project but are useful for exploring time series data.
conda install -c conda-forge sktime  
conda install -c conda-forge aeon

# Or readers can make new environment from our environment. 
# But different platforms may have their own platform-specific packages that may cause error if importing `environments.yml` directly.
conda env create --name envname --file=environments.yml
```

# Project Structure
<!-- https://stackoverflow.com/questions/23989232/is-there-a-way-to-represent-a-directory-tree-in-a-github-readme-md -->
- Important folders and files in this repository are listed as belows:
- Readers should visit the files in this order: `main.ipynb` -> `data-exploration.ipynb` -> `querying.ipynb`.
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
- Information of the relevant paper will be uploaded here after paper acceptance.

# Figures in the Paper
- Figures 1, 3, 4, 5, 6 can be found in `../main.ipynb`.
- Figures 7 (its raw figures), 8 can be found in `../data-exploration.ipynb`.

# Contacts
- It will be updated after paper acceptance.