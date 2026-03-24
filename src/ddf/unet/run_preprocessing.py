#!/net/software/local/python/3.7/bin/python3
# Plan and preprocess starting point.

import argparse
from batchgenerators.utilities.file_and_folder_operations import load_json
from preprocessing.main import main


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=True, help='json file with the configuration of the data needed to perform augmentation')
    args = parser.parse_args()
    config_prep = args.config
    config_data = load_json(config_prep)

    main(config_data)
