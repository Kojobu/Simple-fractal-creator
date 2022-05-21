import numpy as np
import matplotlib.pyplot as plt
import cupy as cp # if no nvidia gpu change this to: 'import numpy as cp' without '
import matplotlib.animation as animation
import matplotlib.cm as cm
from numba import jit
import matplotlib
matplotlib.use('agg')

def exp_gpu(z,c,iterations):
    for _ in range(iterations):
        z = cp.exp(z)+c
    return cp.isfinite(z)

def mandel_gpu(q,c,iterations):
    m = c
    for _ in range(iterations):
        m = cp.power(m,2)+c
        q[cp.abs(m)<=2] += 1
    return cp.log(q+1)

res = 2000 # change according to your gpu-mem
frames = []
#set start
r = 0.25 
theta = cp.double(0.99)*np.pi

x = r*cp.cos(theta)-1
y = r*cp.sin(theta)
num = np.double(4) # set starting range
#set zoom fac
zoom_fac = 0.98
# change this if images wiggly
correction_aggression = 0.001
# change this to determine the divergence level for iterative zoom corrector (the lower the value, the higher the level of divergence)
divergence = 2.9
datatype = cp.complex128 # leave this unchanged if you don't know what it does
dpi = 80
height, width = (res,res)
figsize = width / float(dpi), height / float(dpi)
for i in np.arange(1,9901):
    d = 1/(1+1/(i*correction_aggression))
    iterations = 150+int(i/10)
    z0 = cp.zeros((res,res),dtype=float)
    o, t = cp.meshgrid(cp.linspace(-num,num,res).astype(datatype)+x,cp.linspace(-num,num,res).astype(datatype)*1j+y*1j) # change offset here for poi
    dat = o + t
    z = mandel_gpu(z0,dat,iterations)
    anum = num/res
    # iterative zoom corrector
    unique = cp.unique(z).astype(datatype)
    coord = cp.where(z.T == unique[int(unique.size/divergence)])
    coord = cp.vstack((coord[0],coord[1]))*2*anum
    new_coord = cp.argmin(cp.linalg.norm(coord - num,axis=0))
    new_coord = (coord[:,new_coord] - num)
    x += new_coord[0]*d*0.25
    y += new_coord[1]*d*0.25
    num *= zoom_fac
    # build image
    fig = plt.figure(figsize=figsize, frameon=False)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    ax.imshow(z.get())
    fig.savefig(f'./img/file{str(i).zfill(4)}.png',dpi=dpi)
    if i%99==0: print(i//90)
    # prevent memory leak
    fig.clf()
    plt.close(fig)
    #plt.pause(.01)
