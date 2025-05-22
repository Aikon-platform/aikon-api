# Line Predictor

## Setup

In order to use line predictor for extraction, you first need to compile CUDA operators

1. Set CUDA_HOME (usually located either in `/usr/local/cuda` or `/usr/lib/cuda`):
    ```bash
    echo $CUDA_HOME  # if already defined, go to next step

    # find CUDA version with (pay attention to version mismatches)
    nvcc --version
    nvidia-smi

    # set CUDA_HOME
    export CUDA_HOME=<path/to/cuda>
    # make sure nvcc version matches selected CUDA_HOME
    export PATH=$CUDA_HOME/bin:$PATH
    export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
    ```
2. Activate python environment in the `api/` root folder
    ```bash
    source venv/bin/activate
    cd app/regions/lib/line_predictor
    ```
3. Inside `line_predictor/` folder, run
    ```bash
    # Compile CUDA operators
    python ./dino/ops/setup.py build install
    # Unit test => could output an outofmemory error
    python ./dino/ops/test.py
    ```

### Troubleshooting

Versions setup know to work:
```bash
$ nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2024 NVIDIA Corporation
Built on Thu_Mar_28_02:18:24_PDT_2024
Cuda compilation tools, release 12.4, V12.4.131
Build cuda_12.4.r12.4/compiler.34097967_0

$ python -c "import torch; print(torch.version)"
2.6.0+cu124

$ python -c "import torchvision; print(torchvision.version)"
0.21.0+cu124
```

## Copyright

> ### Line Predictor
> Copyright (c) 2024 Raphaël Baena (Imagine team - LIGM)
> Licensed under the Apache License, Version 2.0
> Copied from LinePredictor (https://github.com/raphael-baena/LinePredictor)
