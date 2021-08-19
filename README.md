# BUSA90501_ML_2021
Machine learning workshop materials . 

Use the link below to run Jupyter notebooks in this repository directly on the cloud via binder](https://mybinder.org/) with a server hosted by [JupyterHub](https://jupyterhub.readthedocs.io/en/latest/). 
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/edmundlth/BUSA90501_ML_2021/main)

# Local installation Guide
 1. Clone repository 
 ```
 git clone https://github.com/edmundlth/BUSA90501_ML_2021.git
 ```
 2. Create Python virtual environment and install required packages specified in `Pipfile.lock` using [pipenv](https://pipenv.pypa.io/en/latest/)
 ```
 cd AMSIWS2021_neural_network_workshop
 pipenv install 
 ```
 3. Run `jupyter` within virtual environment
 ```
 pipenv run jupyter notebook
 ```