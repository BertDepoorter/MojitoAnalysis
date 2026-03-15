# Mojito analysis  
Code for matching `fastlisaresponse` to the Mojito simulation. 

The relevant notebook is `FLR_sim.ipynb`. This should be understandable for other people to go through. The file `output/results.md` contains the quantitative results from the end of the notebook for each source. The subfolders in `output/` contain figures for the time domain and frequency domain responses and residuals.  

## Repo content
The repository contains the following relevant files:
- `FLR_sim.ipynb`: the main notebook for the analysis.
- `test.py`: a simple script to test if the GPU is found correctly by FEW
- `lisa_env.def`: the apptainer definition file to build the container image with all dependencies for the notebook.
- `PE_validation.py`: simple PE script to validate the EMRIs in Mojito against the fast template model. 
- `output/`: folder containing the results of the notebook, including the quantitative results in `results.md` and the figures in the subfolders.
- `data/`: folder containing the MCMC chains and Eryn backend files for all sources.


## Reproduction
To reproduce the code: I have added a lisa_env.def script which contains everything needed: pip installable packages and source installations.
If you want to build the container: first make sure that apptainer can talok to the ESA Gitlab registry. 
```console
apptainer registry login -u <username> docker://gitlab.esa.int:4567
```
Here you should authenticate with personal access token

Optional: reset env variable for apptainer build to large storage
```console
$ export APPTAINER_CACHEDIR=<path/to/large/storage>/.apptainer_cache
$ export APPTAINER_TMPDIR=<path/to/large/storage>/apptainer_tmp
```
To build the .sif container image
```console
$ apptainer build --fakeroot lisa_env.sif lisa_env.def
```

That's it to build the container image. To test if the GPU is found correctly by FEW:
```console
$ apptainer exec --nv lisa_env.sif python3 test.py
```
