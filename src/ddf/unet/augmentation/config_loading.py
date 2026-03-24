from batchgenerators.utilities.file_and_folder_operations import load_json, join, load_pickle


def load_config_file(file_name='config_gen.json'):
    """
    config_gen.json file can be edited to set paths, files names, fold, whether to use or not deep supervision etc.
    :param file_name: default: config_gen.json
    :param file_path: path to the file, by default it would be current working directory.
    :return: dictionary with configuration data.
    """
    config_data = load_json(file_name)
    try:
        load_pickle(config_data['plans_file_path'])
    except FileNotFoundError as e:
        print(e)

    return config_data


if __name__ == '__main__':
    print(load_config_file())
