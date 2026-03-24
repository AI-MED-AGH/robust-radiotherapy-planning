import importlib
import pkgutil
from batchgenerators.utilities.file_and_folder_operations import join


def recursive_find_python_class(folder, trainer_name, current_module):
    """
    Searches for the appropriate class, like e.g. "class ExperimentPlanner3D_v21(ExperimentPlanner)", in folders based
    on trainer_name (e.g. planner_name3d='ExperimentPlanner3D_v21') using module pkgutil.
    :param folder: list of folders in which search will be performed.
    :param trainer_name: e.g. planner_name3d='ExperimentPlanner3D_v21'
    :param current_module: e.g. experiment_planning folder from preprocessing folder.
    :return:
    """
    tr = None
    # MS files and folders in ./experiment_planning, if ispkg is True it means that it is folder, else it is file.
    for importer, modname, ispkg in pkgutil.iter_modules(folder):
        # print(modname, ispkg)
        if not ispkg:
            # MS The importlib module includes functions that implement Python’s import mechanism for loading code in
            # packages and modules. It is one access point to importing modules dynamically, and useful in some cases
            # where the name of the module that needs to be imported is unknown when the code is written (for example,
            # for plugins or extensions to an application).
            m = importlib.import_module(current_module + "." + modname)
            if hasattr(m, trainer_name):  # MS If module has attribute (field in class) with same name as in trainer_name
                tr = getattr(m, trainer_name)  # MS value of that attribute will be loaded.
                break  # MS loads class ExperimentPlanner3D_v21 from experiment_planner_baseline_3DUNet_v21.py
    # MS if tr was not found, search in sub folders.
    if tr is None:
        for importer, modname, ispkg in pkgutil.iter_modules(folder):
            if ispkg:
                next_current_module = current_module + "." + modname
                tr = recursive_find_python_class([join(folder[0], modname)], trainer_name, current_module=next_current_module)
            if tr is not None:
                break

    return tr
