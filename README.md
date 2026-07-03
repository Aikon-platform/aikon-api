# API

This repository contains the code for the worker API.

## Development

```bash
python install.py --mode dev    # generates .env, creates the venv (uv sync), inits submodules
python run.py                   # starts flask + dramatiq, Ctrl+C to stop
```

Or manually:

```bash
uv sync --group=dev
uv run flask --app app.main run --debug
uv run dramatiq app.main -p 1 -t 1
```

Configuration lives in `.env`, generated from [.env.template](.env.template) by `install.py`.
When the api is installed as part of the full AIKON bundle, the values shared with the front
(modules, ports, proxies, data folder) are taken from the root `.env` and must not be edited here.

Update the submodule code with:

```bash
git submodule update --remote
```

### Adding a new demo

1. Create a new demo folder, containing at least a `__init__.py`, `routes.py` and `tasks.py` files
2. Add relevant variables in [`.env.template`](.env.template) and regenerate [`.env`](.env) with `python install.py`
3. If necessary, configure a new xaccel redirection in the [nginx configuration file](docker/nginx.conf.template)
4. Add the demo name (i.e. folder name) to the list `INSTALLED_APPS` in [`.env`](.env)

### Updating the documentation

```bash
uv pip install sphinx furo
cd docs
uv run make html
```

## Deployment

The api runs as a single docker container (nginx + flask + dramatiq under supervisord), standalone or alongside the front:

```bash
python install.py --mode prod   # prompts, renders the docker confs, builds and starts
python run.py build             # rebuild the image + restart (needed after any .env change)
python run.py logs
python run.py down
```

GPU support: set `DEVICE_NB` and `CUDA_HOME` in `.env` (leave `DEVICE_NB` empty for cpu-only).
When installed as part of the bundle in `local` mode, the container joins the front compose network and is reachable by the web app at `http://api:<API_PORT>`.

## Citation

If you find [this work](https://link.springer.com/article/10.1007/s10032-026-00581-x) useful, please consider citing:

```bibtex
@article{albouy2026aikon,
    title={{AIKON : A Modular Computer Vision Platform for Historical Corpora}},
    author={
        Albouy, Ségolène and
        Norindr, Somkeo and
        Kervegan, Paul and
        Aouinti, Fouad and
        Delanaux, Rémy and
        Champenois, Robin and
        Grometto, Clara and
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
