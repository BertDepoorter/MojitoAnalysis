# Mojito analysis  
Code for matching `fastlisaresponse` to the Mojito simulation. 

The relevant notebook is `FLR_sim.ipynb`. This should be understandable for other people to go through. The file `output/results.md` contains the quantitative results from the end of the notebook for each source. The subfolders in `output/` contain figures for the time domain and frequency domain responses and residuals.  

## Reproduction
To reproduce the code: I have added a lisa_env.def script which contains everything needed: pip installable packages and source installations.
If you want to build the container:
```console
# let apptainer talk to the esa gitlab
apptainer registry login -u <username> docker://gitlab.esa.int:4567
# Here you should authenticate with personal access token

# Optional: reset env variable for apptainer build to large storage
export APPTAINER_CACHEDIR=/scratch/project_2004833/depoorter/.apptainer_cache
export APPTAINER_TMPDIR=/scratch/project_2004833/depoorter/apptainer_tmp

# build .sif image
apptainer build --fakeroot lisa_env.sif lisa_env.def
```

That's it to build the container image. To test if the GPU is found correctly:
```console
# test if GPU is found correctly. 
apptainer exec --nv lisa_env.sif python3 -c '
import few
for backend in ["cpu", "cuda11x", "cuda12x", "cuda", "gpu"]:
    print(f" - Backend '{backend}': {"available" if few.has_backend(backend) else "unavailable"}")
'
```
