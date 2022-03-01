# Colorado
### Plot point-clouds, volumes and meshes in Python Notebooks.

Content:
> [Introduction](#intro)

> [Install](#install)

> [Get Started](#getstarted)

---

<a id="intro"></a>
## Introduction

Colorado plots 3D data, such as point-clouds, volumes and meshes. It uses __plotly__, hence can be easily integrated in a Jupyter Notebook.
Colordao's `draw` function identifies the data type/format and choses the best representation for you.

Colorado can draw numpy arrays, or any custom object. 


```python
import colorado as cld
from soma import aims
meshR = aims.read('../docs/tutorial/data/subject01_Rhemi.mesh')
meshL = aims.read('../docs/tutorial/data/subject01_Lhemi.mesh')
#cld.draw([meshL, meshR])
```

![image](../docs/tutorial/markdown/readme/Readme_files/brain.jpg)

The `colorado` package is simply glue between `numpy` and `plotly`, but it also allows to plot aims objects (buckets volumes and meshes) in a [Brainvisa](https://brainvisa.info/web/) environment.

---

<a id="install"></a>
## Install
1. clone this repo

2. install the module:

```{bash}
$ cd colorado
$ pip install .
```

The following steps are necessary to plot in Jupyter:

3. install node and npm (takes minutes). Run the `instal_node.sh` script:
```
$ ./install_node.sh
```

4. Install the plotly plug-in in [Jupyter notebook](https://plotly.com/python/getting-started/#jupyter-notebook-support) or [Jupyter-lab](https://plotly.com/python/getting-started/#jupyterlab-support) (instructions in the links).

### Install with Brainvisa
You might want to install colorado in an environment that already has pyAims to draw aims entites such as `buckets` and `volumes`.
To do so, isntall colorado from a Brainvisa shell. In Brainvisa ()>=5), you can enter a new shell with the command `bv bash`.

**NOTE** on running pip in `bv bash`:
    To make sure you are installing python modules in the right python environment (namely that of Brainvisa) use `python3 -m pip` instead of `pip`




### Resources
* The [interactive version of this notebook](https://neurospin.github.io/colorado/tutorial/markdown/readme/Readme.html)
* The example data used in this document are available in `docs/tutorial/data`
* [documentation](https://neurospin.github.io/colorado/build/html/colorado.html#module-colorado)

---

<a id="intro"></a>
## Get started
Colorado' `draw` function choses the best representation according to the type of the first calling argument.
The object that can be drawn are:
- numpy arrays
- aims buckets volumes and meshes
- any objects that implements the specific `__draw_with_colorado__` method (returning a plotly graphic object)

The `draw()` function can plot numpy arrays, in this case the type of object is inferred by the array dimensions

### Point-clouds
arrays of shape = (N,3) are interpreted as point-clouds (i.e. lists of coordinates) and plotted as scatter plots


```python
import numpy as np

# plot a point cloud
pc = np.load("../docs/tutorial/data/numpy_pc.npy")
#cld.draw(pc)
```

![image](../docs/tutorial/markdown/readme/Readme_files/point_cloud.png)

### Volumes
Arrays of shape = (L,N,M) are interpreted as 3D volumetric images.
The positive valued voxels are displayed in a scatter plot.


```python
vol = aims.read('../docs/tutorial/data/subject01.nii')
#cld.draw(vol, downsample=2, max_points=5000, th_min=950, th_max=None)
# - downsample        downsample the voxels in the volume
# - max_points        number of randomly sampled points to plot (low => fast)
# - th_min            voxels below this value will not be plotted
# - th_max            voxels above this value will not be plotted
```

![image](../docs/tutorial/markdown/readme/Readme_files/volume.png)

### Meshes
eshes are defined by sets of vertices and polygons.
colorado has a handy `SimpleMesh` class that can be used to draw meshes


```python
vertices = np.load('../docs/tutorial/data/mesh_vertices.npy')
polygons = np.load('../docs/tutorial/data/mesh_polygons.npy')

mesh = cld.drawables.SimpleMesh(vertices, polygons)
#cld.draw(mesh)
```

![image](../docs/tutorial/markdown/readme/Readme_files/mesh.png)

## multi-plot
iterables and dictionnaries can also be plot by the draw function


```python
#cld.draw([pc,mesh])
```

![image](../docs/tutorial/markdown/readme/Readme_files/two.png)

the `draw` function returns a plotly window, that can be reused after as an argument to incrementally add entites to the plot


```python
fig = cld.draw(pc)
#cld.draw(mesh, fig = fig, name = "mesh")
```

![image](../docs/tutorial/markdown/readme/Readme_files/two.png)

### Plotting options and Makeup

Any optional argument passed to `draw()` which is not defined in the prototype, is directly passed to the underlying Plotly function.
For example the `marker` argument can be used to change size, shape and color of the points in a bucket plot.
See "3D Scatter Plot" in Plotly's documentation for a complete definition of all available options.


```python
#cld.draw(a, marker=dict(color='red'))
```

![image](../docs/tutorial/markdown/readme/Readme_files/red.png)
