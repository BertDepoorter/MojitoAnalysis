# Mojito analysis  
Code for matching `fastlisaresponse` to the Mojito simulation. 

The relevant notebook is `FLR_sim.ipynb`. This should be understandable for other people to go through. The file `output/results.md` contains the quantitative results from the end of the notebook for each source. The subfolders in `output/` contain figures for the time domain and frequency domain responses and residuals.  

## Reproduction
To reproduce the code: I have added a lisa_env.def script which contains everything needed: pip installable packages and source installations.
If you want to build the container:
```console
# build .sif image
apptainer build --fakeroot lisa_env.sif lisa_env.def

# test if GPU is found correctly. 
apptainer exec --nv lisa_env.sif python3 -c
```
