
from batchgenerators.utilities.file_and_folder_operations import load_json, join, load_pickle


def load_config_file(file_name='config_prep.json'):
    """
    config_prep.json file can be edited to set paths, files names, fold, whether to use or not deep supervision etc.
    :param file_name: default: config_prep.json
    :return: dictionary with configuration data.
    """
    config_data = load_json(file_name)

    return config_data


if __name__ == '__main__':
    print(load_config_file())
    print(type(load_config_file()['task_ids']))

