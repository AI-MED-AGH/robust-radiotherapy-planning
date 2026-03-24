import os
import shutil
from preprocessing.crop import crop, create_lists_from_splitted_dataset
from preprocessing.experiment_planning.DatasetAnalyzer import DatasetAnalyzer
from batchgenerators.utilities.file_and_folder_operations import join, load_json, maybe_mkdir_p
from preprocessing.paths import convert_id_to_task_name
from preprocessing.plan import recursive_find_python_class
from preprocessing.verify import verify_dataset_integrity
from preprocessing.paths import set_paths

RESAMPLING_SEPARATE_Z_ANISO_THRESHOLD = 3  # determines what threshold to use for resampling the low resolution axis


def main(config_data):
    # MS Dropped: Terminal commands and information (argparse).
    # List of integers belonging to the task ids you wish to run experiment planning and preprocessing for.
    # Each of these ids must, have a matching folder 'TaskXXX_' in the raw data folder
    #task_ids = config_data['task_ids']  # task_ids = ['7']
    dont_run_preprocessing = config_data['dont_run_preprocessing']  # dont_run_preprocessing = False
    # tl - number of processes used for preprocessing the low resolution data for the 3D low
    # resolution U-Net. This can be larger than -tf. Don't overdo it or you will run out of RAM
    tl = config_data['tl']  # tl = 8
    # tf - number of processes used for preprocessing the full resolution data of the 2D U-Net and
    # 3D U-Net. Don't overdo it or you will run out of RAM
    tf = config_data['tf']  # tf = 8
    # ExperimentPlanner3D_v21 - name of the ExperimentPlanner class for the full resolution 3D U-Net and U-Net cascade.
    # Default is ExperimentPlanner3D_v21. Can be 'None', in which case these U-Nets will not be configured
    planner_name3d = config_data['planner_name3d']  # planner_name3d = 'ExperimentPlanner3D_v21'
    # ExperimentPlanner2D_v21 - name of the ExperimentPlanner class for the 2D U-Net. Default is
    # ExperimentPlanner2D_v21. Can be 'None', in which case this U-Net will not be configured
    planner_name2d = config_data['planner_name2d']  # planner_name2d = 'ExperimentPlanner2D_v21'
    # set this flag to check the dataset integrity. This is useful and should be done once for each dataset!
    verify_dataset_integrity_flag = config_data['verify_dataset_integrity_flag']  # verify_dataset_integrity_flag = True

    default_num_threads = config_data['default_num_threads']

    nnUNet_raw_data, nnUNet_cropped_data, preprocessing_output_dir = set_paths(config_data)

    if planner_name3d == "None":
        planner_name3d = None
    if planner_name2d == "None":
        planner_name2d = None

    if verify_dataset_integrity_flag:
        verify_dataset_integrity(nnUNet_raw_data, default_num_threads)

    crop(False, tf, nnUNet_raw_data, nnUNet_cropped_data)

        # MS Task name and some folders creations.
    
    # MS: Searches for the appropriate class, like e.g. "class ExperimentPlanner3D_v21(ExperimentPlanner)", in folders
    # based on trainer_name (e.g. planner_name3d='ExperimentPlanner3D_v21') using module pkgutil.
    search_in = join('preprocessing', "experiment_planning")  # MS where to search.

    if planner_name3d is not None:
        planner_3d = recursive_find_python_class([search_in], planner_name3d, current_module="preprocessing.experiment_planning")
        if planner_3d is None:
            raise RuntimeError("Could not find the Planner class %s. Make sure it is located somewhere in "
                               "experiment_planning" % planner_name3d)
    else:
        planner_3d = None

    # MS same as above but with 2D.
    if planner_name2d is not None:
        planner_2d = recursive_find_python_class([search_in], planner_name2d, current_module="preprocessing.experiment_planning")
        if planner_2d is None:
            raise RuntimeError("Could not find the Planner class %s. Make sure it is located somewhere in "
                               "nnunet.experiment_planning" % planner_name2d)
    else:
        planner_2d = None
    # MS: End of "Searches for the appropriate class..."

    cropped_out_dir = nnUNet_cropped_data
    preprocessing_output_dir_this_task = preprocessing_output_dir
    splitted_4d_output_dir_task = nnUNet_raw_data
    lists, modalities = create_lists_from_splitted_dataset(splitted_4d_output_dir_task)

    # we need to figure out if we need the intensity properties. We collect them only if one of the modalities is CT
    # intensity - how vivid it is. Low - grayish, high - bright.
    dataset_json = load_json(join(cropped_out_dir, 'dataset.json'))
    modalities = list(dataset_json["modality"].values())
    collect_intensityproperties = True if (("CT" in modalities) or ("ct" in modalities)) else False

    print('creatinf fingerprints')

    dataset_analyzer = DatasetAnalyzer(cropped_out_dir, overwrite=False, num_processes=tf)  # this class creates the fingerprint MS __init__ from DatasetAnalyzer - class fields setting.
    _ = dataset_analyzer.analyze_dataset(collect_intensityproperties)  # this will write output files that will be used by the ExperimentPlanner

    # Creating and copying to the nnUNet_preprocessed folder.
    maybe_mkdir_p(preprocessing_output_dir_this_task)
    shutil.copy(join(cropped_out_dir, "dataset_properties.pkl"), preprocessing_output_dir_this_task)
    shutil.copy(join(nnUNet_raw_data, "dataset.json"), preprocessing_output_dir_this_task)

    threads = (tl, tf)

    print("number of threads: ", threads, "\n")

    if planner_3d is not None:
        exp_planner = planner_3d(cropped_out_dir, preprocessing_output_dir_this_task)
        exp_planner.plan_experiment()
        if not dont_run_preprocessing:  # double negative, yooo
            exp_planner.run_preprocessing(threads, default_num_threads)  # Among others things, copying to the nnUNet_preprocessed folder.

    if planner_2d is not None:
        exp_planner = planner_2d(cropped_out_dir, preprocessing_output_dir_this_task)
        exp_planner.plan_experiment()
        if not dont_run_preprocessing:  # double negative, yooo
            exp_planner.run_preprocessing(threads, default_num_threads)  # Among others things, copying to the nnUNet_preprocessed folder.

