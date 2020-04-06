import pickle
from skimage import measure
from skimage import color, io, transform
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os
from datetime import datetime

path = r"\\169.254.39.157\Shared_Folder\Webcam_Test"
crop, start_time, filetype, angle = eval(open(path + "/data.txt", "r").read()).values()
start_time = datetime.strptime(start_time, "%Y%m%d%H%M%S")
entropy_data = []
time = []
thinning_constant = 5
n = 0
for file in tqdm(os.listdir(path)):
    n += 1
    if file.endswith(filetype) and n % thinning_constant == 0:
        im = io.imread(path + "\\" + file)
        im = transform.rotate(im, float(angle))
        im = im[crop[0]:crop[1], crop[2]:crop[3]]
        time.append((datetime.strptime(file[:-len(filetype)], "%Y%m%d%H%M%S") - start_time).total_seconds() / (60 * 60))
        entropy_data.append(measure.shannon_entropy(im))


plt.scatter(time, entropy_data)
plt.xlabel("Shannon Entropy")
plt.ylabel("Time (hours)")
plt.show()
