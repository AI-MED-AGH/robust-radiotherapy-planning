# Folder paths.

from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p, join, subdirs
import numpy as np

# MS Dropped: all things related to training
# MS Dropped(left commented): convert_task_name_to_id

# my_output_identifier = "nnUNet"
default_plans_identifier = "nnUNetPlansv2.1"  # MS: prefix of the plan file.
default_data_identifier = 'nnUNetData_plans_v2.1'  # MS: prefix of the preprocessed data.


def set_paths(config_data):
    """
    obtains paths from configuration file (config_prep.json)
    :param config_data: data read from config_prep.json file
    :return: nnUNet_raw_data, nnUNet_cropped_data, preprocessing_output_dir
    """
    
    nnUNet_raw_data = config_data['folder_with_raw_data']
    nnUNet_cropped_data = config_data["folder_with_cropped_data"]
    preprocessing_output_dir = config_data['preprocessing_output_dir']
    maybe_mkdir_p(nnUNet_raw_data)
    maybe_mkdir_p(nnUNet_cropped_data)

    if preprocessing_output_dir is not None:
        maybe_mkdir_p(preprocessing_output_dir)
    else:
        print("nnUNet_preprocessed is not defined and nnU-Net can not be used for preprocessing")
        preprocessing_output_dir = None

    return nnUNet_raw_data, nnUNet_cropped_data, preprocessing_output_dir


def convert_id_to_task_name(task_id: int, nnUNet_raw_data, nnUNet_cropped_data, preprocessing_output_dir):
    """
    :param task_id: e. g. '7' for Task007_Liver
    :return: the sorted unique task names ('Task007_Liver')
    """
    startswith = "Task%03.0d" % task_id
    if preprocessing_output_dir is not None:
        candidates_preprocessed = subdirs(preprocessing_output_dir, prefix=startswith, join=False)
    else:
        candidates_preprocessed = []

    if nnUNet_raw_data is not None:
        candidates_raw = subdirs(nnUNet_raw_data, prefix=startswith, join=False)
        #print('candidates_raw', candidates_raw)
    else:
        candidates_raw = []

    if nnUNet_cropped_data is not None:
        candidates_cropped = subdirs(nnUNet_cropped_data, prefix=startswith, join=False)
    else:
        candidates_cropped = []

    all_candidates = candidates_cropped + candidates_preprocessed + candidates_raw
    #print(all_candidates)
    unique_candidates = np.unique(all_candidates)
    #print(unique_candidates)
    if len(unique_candidates) > 1:
        raise RuntimeError("More than one task name found for task id %d. Please correct that. (I looked in the "
                           "following folders:\n%s\n%s\n%s" % (task_id, nnUNet_raw_data, preprocessing_output_dir,
                                                               nnUNet_cropped_data))
    #if len(candidates_cropped) == 0:
    if len(candidates_raw) == 0:
        raise RuntimeError("Could not find a task with the ID %d. Make sure the requested task ID exists and that "
                           "paths to the raw and preprocessed data are correct." % (task_id,))

    return unique_candidates[0]


# def convert_task_name_to_id(task_name: str):
#     assert task_name.startswith("Task")
#     task_id = int(task_name[4:7])
#     return task_id
