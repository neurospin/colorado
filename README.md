# Colorado

## Plot pyAims entities in Python Notebooks.

```{python}
import colorado as cld
meshR = aims.read('data/subject01_Rhemi.mesh')
meshL = aims.read('data/subject01_Lhemi.mesh')
cld.draw([meshL, meshR])
```

![image](./docs/images/brain.jpg)

The `colorado` module is an interface between `aims` and `plotly`.

The functions implemented in this module allow to plot aims objects (buckets volumes and meshes) with plotly.
It can be used, for example, inside jupyter notebooks.

### Install
Most probably, you want to install colorado in an environment that already has aims.
In Brainvisa 5, you can enter the environment with the command `bv bash`. Then:

0. **NOTE** on running pip in `bv bash`:

    To make sure you are installing python modules in the right python environment (namely that of brainvisa) use `python -m pip` instead of `pip`

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

### Resources
* [simple example](https://neurospin.github.io/colorado/tutorial/tutorial.html) The example data are in `docs/tutorial/data`
* [documentation](https://neurospin.github.io/colorado/build/html/colorado.html#module-colorado)
