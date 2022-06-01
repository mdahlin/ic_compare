#!/bin/bash

echo "activating venv"
source .env/bin/activate

echo "building rust bindings"
cd rust && maturin develop --release
