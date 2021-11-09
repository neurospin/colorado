import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(name='Colorado',
                 version='0.2.0',
                 description="A plotly interface for pyAims (brain images)",
                 author='Marco Pascucci',
                 long_description=long_description,
                 long_description_content_type="text/markdown",
                 author_email='marpas.paris@gmail.com',
                 url='',
                 packages=['colorado'],
                 install_requires=['plotly', 'numpy', 'dico_toolbox'],
                 classifiers=[
                     "Programming Language :: Python",
                     "License :: OSI Approved :: GNU General Public License v2 (GPLv2)",
                     "Operating System :: OS Independent",
                 ]
                 )
