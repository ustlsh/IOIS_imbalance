import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image


npz_file = '/home/slidm/OCTA/Awesome-Backbones/results/64cs_rotation_onlinedm_0.2/samples_1984x64x64x3_epoch130.npz'
data = np.load(npz_file)

# arr_0: [n, 256, 256, 3]; arr_1 = [n,]
print(data['arr_0'].shape, data['arr_1'].shape)

fig = plt.figure(figsize=(40, 100))
columns = 20
rows = int(data['arr_0'].shape[0]/columns)
for i in range(1, columns*rows +1):
    img = data['arr_0'][i-1]
    pil_image = Image.fromarray(img)
    print(pil_image.size)
    print(img.min(), img.max())
    label = data['arr_1'][i-1]
    fig.add_subplot(rows, columns, i)
    plt.imshow(img)
    plt.title(str(label))
plt.savefig('/home/slidm/OCTA/Awesome-Backbones/results/64cs_rotation_onlinedm_0.2/samples_1984x64x64x3_epoch130.png')





