import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image


npz_file = '/home/slidm/OCTA/Awesome-Backbones/results/64cs_flip_rot_rc_onlinedm_0.2_ddim50_lrstep60_new/samples_1984x64x64x3_epoch146.npz'
data = np.load(npz_file)

save_dir = "/home/slidm/OCTA/Awesome-Backbones/results/64cs_flip_rot_rc_onlinedm_0.2_ddim50_lrstep60_new/epoch146"

if not os.path.exists(save_dir):
    # Create the directory
    os.makedirs(save_dir)

# arr_0: [n, 256, 256, 3]; arr_1 = [n,]
print(data['arr_0'].shape, data['arr_1'].shape)


for i in range(1984):
    img = data['arr_0'][i]
    pil_image = Image.fromarray(img)
    print(pil_image.size)
    print(img.min(), img.max())
    label = data['arr_1'][i]
    filename = str(label)+"_"+str(i).zfill(4)+".jpg"
    pil_image.save(os.path.join(save_dir, filename))
    #fig.add_subplot(rows, columns, i)
    #plt.imshow(img)
    #plt.title(str(label))
#plt.savefig('/home/slidm/OCTA/Awesome-Backbones/results/64cs_rotation_onlinedm_0.2/samples_1984x64x64x3_epoch130.png')





