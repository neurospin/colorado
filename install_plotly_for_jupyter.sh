#!/bin/bash

DOWNLOAD_PATH="$HOME/.tmp/node-latest-install"
NODE_URL="https://nodejs.org/dist/node-latest.tar.gz"
NODE_FILE="node-latest.tar.gz"

echo 'export PATH=$HOME/.local/bin:$PATH' >> ~/.bashrc
. ~/.bashrc
mkdir -p ~/.local/bin
mkdir -p "$DOWNLOAD_PATH"
cd "$DOWNLOAD_PATH"

wget -Nc "$NODE_URL"
# unzip
tar -xf "$NODE_FILE" --strip-components=1
# Install node
./configure --prefix="~/.local"
make -j4
make  install

# Download the NPM install script and run it
wget -c "https://www.npmjs.org/install.sh" | sh

# install plotpy for jupyter
jupyter-labextension install jupyterlab-plotly@4.14.3
