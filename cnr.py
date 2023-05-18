import numpy as np
from PIL import Image
from glob import glob

dir = "/autofs/cluster/octdata2/users/epc28/ml_vesselseg/vesselsynth/data/exp0001/volumes/RandomGammaNoiseTransform_mean-1_var-0.83/tiffs"


avg_cnr = []

for file in glob(f'{dir}/*'):

    img = Image.open(file)
    arr = np.array(img)

    wm = arr[arr <= 0.20]
    gm = arr[arr > 0.20]

    #print(wm.mean())
    #print(wm.var())

    cnr = abs(wm.mean() - gm.mean()) / (gm.var() + wm.var())**(1/2)
    avg_cnr.append(cnr)
    #print(cnr)

    #print(arr)

print(np.array(avg_cnr).mean())