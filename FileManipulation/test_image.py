import numpy as np
from matplotlib import pyplot as plt

img_list = []
with open("Tadaki+20_S.dat", "r", encoding="UTF-8") as f:
    for i in range(4096):
        img_list.append(f.readline())

img = np.array(img_list, dtype="float64")
img = np.reshape(img, (64, 64))

plt.imshow(img, cmap="gray")
plt.savefig("firstimage.png", format="png")
