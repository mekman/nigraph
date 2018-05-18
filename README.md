[![Build Status](https://travis-ci.org/mekman/nigraph.svg?branch=master)](https://travis-ci.org/mekman/nigraph)
[![Coverage Status](https://coveralls.io/repos/github/mekman/nigraph/badge.svg?branch=master)](https://coveralls.io/github/mekman/nigraph?branch=master)

**Nigraph** is a Python module for graph analyses on NeuroImaging data.

It is especially tailored towards the analysis of functional (**fMRI/MEG/EEG**) and structural (**DTI/DWI**) brain imaging data.

Features include:
- construction of ``static`` and ``dynamic`` graphs from imaging data
- extensive set of ``metrics`` to quantify brain networks and communities
- ``statistical comparison`` of graphs including ``complex network decoding`` *(Ekman et al., 2012)*

### Quick-start

This code snippet shows how to construct a network graph from a resting-state fMRI time-series and calculate the weighted, local betweenness_centrality:


```shell
$ python
```
```python
>>> import nigraph as nig
>>> timeseries = nig.load_mri('rest.nii.gz', 'brain_mask.nii.gz')
>>> adjacency = nig.adj_static(timeseries)
>>> adjacency_thr = nig.thresholding_abs(adjacency, thr=0.3)
>>> bc = nig.betweenness_centrality(adjacency_thr)
```

### Installation

Currently this is only available through GitHub. **Nigraph** will run under Linux and Mac OS X, but not under Windows.

    pip install git+https://github.com/mekman/nigraph.git --upgrade

### Citation

If you use the **Nigraph** for connectivity-based decoding please cite::

    @article{Ekman09102012,
    author = {Ekman, Matthias and Derrfuss, Jan and Tittgemeyer, Marc and Fiebach, Christian J.},
    title = {Predicting errors from reconfiguration patterns in human brain networks},
    volume = {109},
    number = {41},
    pages = {16714-16719},
    year = {2012},
    doi = {10.1073/pnas.1207523109},
    URL = {http://www.pnas.org/content/109/41/16714.abstract},
    eprint = {http://www.pnas.org/content/109/41/16714.full.pdf+html},
    journal = {Proceedings of the National Academy of Sciences}
    }

### License
Copyright (C) 2011-2018 Nigraph Developers

- Matthias Ekman <Matthias.Ekman@gmail.com>
- Charl Linssen <Charl@turingbirds.com>

Distributed with a BSD license (3 clause); see LICENSE.
