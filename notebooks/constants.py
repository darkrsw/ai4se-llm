CONSTANTS = {
    "OUTPUT_ROOT": None,
    "REPO_ROOT": None,
    "MODEL_CACHE_PATH": None,
    "DEV_MEM_MAP": {}
}

import logging
import yaml

def load_constants(const_path):
    # read experiment constants
    logging.info(f"Reading experiment constants from {const_path}")
    with open(const_path, 'r') as f:
        constants = yaml.load(f, Loader=yaml.FullLoader)

    CONSTANTS["OUTPUT_ROOT"] = constants["OUTPUT_ROOT"]
    CONSTANTS["REPO_ROOT"] = constants["REPO_ROOT"]
    ## TODO: cache directory is not used yet.
    CONSTANTS["MODEL_CACHE_PATH"] = constants["MODEL_CACHE_PATH"]

    # this could be null
    if "device_memory_map" in constants:
        CONSTANTS["DEV_MEM_MAP"] = constants["device_memory_map"]

    const_validate()

    return constants

def const_validate():
    assert CONSTANTS["OUTPUT_ROOT"] is not None
    assert CONSTANTS["REPO_ROOT"] is not None
    assert CONSTANTS["MODEL_CACHE_PATH"] is not None



if __name__ == '__main__':
    load_constants("docker_constants.yaml")

    print(CONSTANTS["DEV_MEM_MAP"])

    for key, value in CONSTANTS["DEV_MEM_MAP"].items():
        print(type(key), type(value))

