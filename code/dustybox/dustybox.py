#!/usr/bin/env python
# coding: utf-8

# # Dusty box analysis
# 
# This notebook contains analysis of the dusty box test for multigrain dust.

# ## Imports

# In[1]:


import importlib
import pathlib
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plonk
from numpy import ndarray
from pandas import DataFrame
from plonk import Simulation


# ## Functions
# 
# Load dustybox analysis functions:
# - `load_data`: to generate Plonk Simulation objects
# - `generate_results`: to perform the analysis
# - `plot_results`: to plot the results

# In[2]:


loader = importlib.machinery.SourceFileLoader('dustybox', 'analysis.py')
dustybox = loader.load_module()


# ## Load data

# In[3]:


root_directory = pathlib.Path('~/runs/multigrain/dustybox').expanduser()
sims = dustybox.load_data(root_directory)


# ## Perform analysis

# In[4]:


dataframes = dict()
for sim in sims:
    name = sim.path.name.split('=')[-1]
    dataframes[name] = dustybox.generate_results(sim)


# ## Plot results

# In[ ]:


ncols, nrows = 2, 2
with plt.style.context('ggplot'):
    fig, axes = plt.subplots(
        ncols=ncols, nrows=nrows, sharex='all', sharey='all', figsize=(12, 12)
    )
    dustybox.plot_results(dataframes, fig, axes)

