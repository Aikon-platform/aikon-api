"""
Configuration files of the API.

It has three submodules:

- `base` : contains the base configuration
- `dev` : contains the development configuration
- `prod` : contains the production configuration
"""

from .base import ENV

TARGET = ENV("TARGET", default="").strip()

if TARGET == "dev":
    from .dev import *
elif TARGET == "prod":
    from .prod import *
else:
    raise ValueError("TARGET environment variable must be either 'dev' or 'prod'")
