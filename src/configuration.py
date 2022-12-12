from pathlib import Path
import munch
import yaml

PARAMS = "params.yaml"
def get_params(params_file=PARAMS):
    with Path(params_file).open("rt") as fh:
        params = munch.munchify(yaml.safe_load(fh.read()))
    return params