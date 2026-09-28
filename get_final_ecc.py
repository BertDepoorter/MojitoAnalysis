from few.trajectory.inspiral import EMRIInspiral
from dotenv import load_dotenv
import numpy as np
import os   
from mojito.download import get_source_params

load_dotenv()
my_password = os.getenv("LISA_CONSORTIUM_KEY")
my_username = os.getenv("LISA_CONSORTIUM_NAME")


# create EMRI inspiral object
traj = EMRIInspiral(func='KerrEccEqFlux')

T = 2.0 # years

e_final = {}
# Load mojito parameter file
for source_index in range(0, 8):    
    params = get_source_params("emri", source_id=source_index, username=my_password, token=my_username)

    traj_params = [
        params['PrimaryMassSSBFrame'],
        params['SecondaryMassSSBFrame'],
        params['PrimarySpinParameter'], #* np.sign(np.cos(params['InclinationAngle'])),
        params['SemiLatusRectum'],
        params['Eccentricity'],
        np.cos(params['InclinationAngle'])
    ]

    t, p, e, xI, Phi_phi, Phi_theta, Phi_r = traj(*traj_params, T=T)

    e_final[source_index] = e[-1]

# print output
for source_index in range(0, 8):
    print(f'Final eccentricities Mojito Light')
    print('-----------------------------')
    print(f"Source {source_index}: Final eccentricity = {e_final[source_index]:.4e}")