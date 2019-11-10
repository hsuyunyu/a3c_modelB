import glob
from PIL import Image
import numpy as np
import os

import matplotlib.animation as animation
import matplotlib.pyplot as plt

path = "attention/1/"  # 待讀取的資料夾
picList = os.listdir(path)
picList.sort()  # 對讀取的路徑進行排序
for filename in picList:
    print(os.path.join(path, filename))
ims = []
fig = plt.figure()
plt.axis('off')

for i in range(len(picList)):
    tmp = Image.open(os.path.join(path, picList[i]))
    im = plt.imshow(tmp)
    ims.append([im])
ani = animation.ArtistAnimation(fig, ims, interval=200)

ani.save(path+'attention.gif', writer='imagemagick', dpi=400)




