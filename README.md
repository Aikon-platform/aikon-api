# API

This folder contains the code for the worker API.

## Development

> Run `bash setup.sh` to execute scripted install

Copy the file `.env.template` to a file `.env`. Change its content to match your setup (especially regarding the paths).

Install redis and python:

```bash
sudo apt-get install redis-server python3-venv python3-dev
```

Initialize submodules:

```bash
git submodule init
git submodule update

# once initiate, update the submodule code with
git submodule update --remote
```

Create a python virtual environment and install the required packages:

```bash
python3 -m venv venv
./venv/bin/pip install -r requirements-dev.txt
```

You can now run the API worker:

```bash
./venv/bin/dramatiq app.main -p 1 -t 1
```

And the server:

```bash
./venv/bin/flask --app app.main run --debug
```

### Adding new demo

1. Create a new demo folder, containing at least a `__init__.py`, `routes.py` and `tasks.py` files
2. Add relevant variables in [`.env.template`](.env.template) and generate the corresponding [`.env`](.env) file
3. If necessary, configure a new xaccel redirection in the [nginx configuration file](docker-confs/nginx.conf)
4. Add the demo name (i.e. folder name) to the list `INSTALLED_APPS` in [`.env`](.env)

### Updating the documentation

You need to install `sphinx` and the `furo` theme:

```bash
./venv/bin/pip install sphinx furo
```

You can then generate the documentation with `make`:

```bash
cd docs
make html
```

## Deployment

See [docker/README.md](docker/README.md) for deployment instructions.

## Citation

If you find this work useful, please consider citing:

```bibtex
@article{albouy2025aikon,
    title={{AIKON : A Modular Computer Vision Platform for Historical Corpora}},
    author={
        Albouy, Ségolène and
        Norindr, Somkeo and
        Kervegan, Paul and
        Aouinti, Fouad and
        Delanaux, Rémy and
        Champenois, Robin and
        Lazaris, Stavros and
        Guilbaud, Alexandre and
        Husson, Matthieu and
        Aubry, Mathieu
    },
    url={https://hal.science/hal-05248250},
    year={2025},
    month={Sep},
    number={hal-05248250},
    journal={HAL Pre-Print},
    keyword={Digital Humanities, Computer Vision, Historical Documents, Visual Analysis},
}
```
