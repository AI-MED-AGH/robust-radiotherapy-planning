import os
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from batchgenerators.utilities.file_and_folder_operations import load_pickle, save_pickle, isfile, join
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter
from batchgenerators.transforms.abstract_transforms import Compose
from batchgenerators.transforms.channel_selection_transforms import DataChannelSelectionTransform, \
    SegChannelSelectionTransform
from batchgenerators.transforms.color_transforms import BrightnessMultiplicativeTransform, BrightnessTransform, \
    ContrastAugmentationTransform, GammaTransform
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform, GaussianBlurTransform
from batchgenerators.transforms.resample_transforms import SimulateLowResolutionTransform
from batchgenerators.transforms.spatial_transforms import SpatialTransform, MirrorTransform
from batchgenerators.transforms.utility_transforms import RemoveLabelTransform, RenameTransform, NumpyToTensor
from augmentation.custom_transforms import Convert3DTo2DTransform, Convert2DTo3DTransform, \
    MaskTransform, ConvertSegmentationToRegionsTransform
from augmentation.pyramid_augmentations import MoveSegAsOneHotToData, \
    ApplyRandomBinaryOperatorTransform, \
    RemoveRandomConnectedComponentFromOneHotEncodingTransform
from augmentation.downsampling import DownsampleSegForDSTransform3, DownsampleSegForDSTransform2
from sklearn.model_selection import KFold
from augmentation.dataset_loading import load_dataset, DataLoader2D, DataLoader3D
import numpy as np
from collections import OrderedDict
from augmentation.config_loading import load_config_file
from augmentation.default_data_augmentation import get_patch_size, default_3D_augmentation_params, default_2D_augmentation_params

NUM_OF_SPLITS = 5

def do_split(dataset, fold, config_data):
    """
    Splits data according to fold parameter.
    :param dataset: Ordered dictionary with images and properties like spacings or size after cropping.
    :param fold: when set to all both tr_keys and val_keys from dataset are used together as one key set.
    Otherwise they are splitted.
    :param config_data: dictionary with configuration settings from config_gen.json
    :return: training and validation dataset.
    """
    # MS: os.path.dirname to drop last folder from config_data['folder_with_preprocessed_data']
    splits_file = join(os.path.dirname(config_data['folder_with_preprocessed_data']), 'splits_final.pkl')
    splits = []
    all_keys_sorted = np.sort(list(dataset.keys()))
    # MS: KFold from sklearn, n_splits from config_gen.json
    kfold = KFold(n_splits = NUM_OF_SPLITS, shuffle=True, random_state=config_data['random_state'])
    for i, (train_idx, test_idx) in enumerate(kfold.split(all_keys_sorted)):
        train_keys = np.array(all_keys_sorted)[train_idx]
        test_keys = np.array(all_keys_sorted)[test_idx]
        splits.append(OrderedDict())
        splits[-1]['train'] = train_keys
        splits[-1]['val'] = test_keys
    save_pickle(splits, splits_file)


    if fold == "all":
        tr_keys = val_keys = list(dataset.keys())
    else:
        tr_keys = splits[fold]['train']
        val_keys = splits[fold]['val']

    tr_keys.sort()
    val_keys.sort()

    dataset_tr = OrderedDict()
    for i in tr_keys:
        dataset_tr[i] = dataset[i]

    dataset_val = OrderedDict()
    for i in val_keys:
        dataset_val[i] = dataset[i]

    print('Using following data for training:',tr_keys,flush=True)
    print('Using following data for validation:',val_keys,flush=True)
    return dataset_tr, dataset_val


def setup_DA_params(threeD, do_dummy_2D_aug, patch_size, use_mask_for_norm):
        """
        Setups data augmentations parameters.
        :param threeD: value 2 or 3 depending on patch size length; patch size is obtained from the plan file.
        :param do_dummy_2D_aug: from plan file plan['plans_per_stage'][stage]['do_dummy_2D_data_aug']
        :param patch_size: patch_size (2 or 3 elements array) obtained from the plan file.
        :param use_mask_for_norm: obtained from the plan file.

        In this version:
        increased roation angle from [-15, 15] to [-30, 30]
        different  scale range, now (0.7, 1.4), was (0.85, 1.25)
        elastic deformation set to False

        :return: dictionary of data augmentation parameters, and basic generator patch size
        """

        if threeD:
            data_aug_params = default_3D_augmentation_params
            data_aug_params['rotation_x'] = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
            data_aug_params['rotation_y'] = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
            data_aug_params['rotation_z'] = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
            if do_dummy_2D_aug:
                data_aug_params["dummy_2D"] = True
                data_aug_params["elastic_deform_alpha"] = \
                    default_2D_augmentation_params["elastic_deform_alpha"]
                data_aug_params["elastic_deform_sigma"] = \
                    default_2D_augmentation_params["elastic_deform_sigma"]
                data_aug_params["rotation_x"] = default_2D_augmentation_params["rotation_x"]
        else:
            do_dummy_2D_aug = False
            if max(patch_size) / min(patch_size) > 1.5:
                default_2D_augmentation_params['rotation_x'] = (-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi)
            data_aug_params = default_2D_augmentation_params
        data_aug_params["mask_was_used_for_normalization"] = use_mask_for_norm

        if do_dummy_2D_aug:
            basic_generator_patch_size = get_patch_size(patch_size[1:],
                                                             data_aug_params['rotation_x'],
                                                             data_aug_params['rotation_y'],
                                                             data_aug_params['rotation_z'],
                                                             data_aug_params['scale_range'])
            basic_generator_patch_size = np.array([patch_size[0]] + list(basic_generator_patch_size))
            patch_size_for_spatialtransform = patch_size[1:]
        else:
            basic_generator_patch_size = get_patch_size(patch_size, data_aug_params['rotation_x'],
                                                             data_aug_params['rotation_y'],
                                                             data_aug_params['rotation_z'],
                                                             data_aug_params['scale_range'])
            patch_size_for_spatialtransform = patch_size

        data_aug_params["scale_range"] = (0.7, 1.4)
        data_aug_params["do_elastic"] = False
        data_aug_params['selected_seg_channels'] = [0]
        data_aug_params['patch_size_for_spatialtransform'] = patch_size_for_spatialtransform

        data_aug_params["num_cached_per_thread"] = 2

        return data_aug_params, basic_generator_patch_size


def get_basic_generators(config_data):
    """
    Reads data (batch_size, patch_size, use_mask_for_norm) from the plan file and generates training and validation
    Data Loaders based on this data.
    :param config_data: dictionary with configuration settings from config_gen.json
    :return: training and validation Data Loaders.
    """
    plan = load_pickle(config_data['plans_file_path'])
    dataset = load_dataset(config_data['folder_with_preprocessed_data'])
    # fold is being used in do_split function, when set to all both tr_keys and val_keys from dataset are used together
    # as one key set. Otherwise they are splitted.
    fold = config_data['fold']
    stage = 0
    # oversample_foreground: half the batch will be forced to contain at least some foreground (equal prob for each of the foreground classes)
    oversample_foreground_percent = 0.0
    pad_all_sides = None
    batch_size = plan['plans_per_stage'][stage]['batch_size']
    try:
        # Obtained from a plan file
        # print(plan['plans_per_stage'][0]['patch_size'])
        patch_size = plan['plans_per_stage'][stage]['patch_size']
        if len(patch_size) == 2:
            threeD = False
        elif len(patch_size) == 3:
            threeD = True
        else:
            raise Exception('Patch size length and threeD, which is derived from patch size length, is not equal to 2 '
                            'or 3')
    except Exception as ex:
        print(ex)
        raise
    use_mask_for_norm = plan['use_mask_for_norm']
    # default_3D_augmentation_params - base for data_aug_params
    data_aug_params, basic_generator_patch_size = setup_DA_params(threeD, plan['plans_per_stage'][stage]['do_dummy_2D_data_aug'], patch_size, use_mask_for_norm)
    dataset_tr, dataset_val = do_split(dataset, fold, config_data)

    if threeD:
        dl_tr = DataLoader3D(dataset_tr, basic_generator_patch_size, patch_size, batch_size,
                             False, oversample_foreground_percent=oversample_foreground_percent,
                             pad_mode="constant", pad_sides=pad_all_sides, memmap_mode='r')
        dl_val = DataLoader3D(dataset_val, patch_size, patch_size, batch_size, False,
                              oversample_foreground_percent=oversample_foreground_percent,
                              pad_mode="constant", pad_sides=pad_all_sides, memmap_mode='r')
    else:
        dl_tr = DataLoader2D(dataset_tr, basic_generator_patch_size, patch_size, batch_size,
                             oversample_foreground_percent=oversample_foreground_percent,
                             pad_mode="constant", pad_sides=pad_all_sides, memmap_mode='r')
        dl_val = DataLoader2D(dataset_val, patch_size, patch_size, batch_size,
                              oversample_foreground_percent=oversample_foreground_percent,
                              pad_mode="constant", pad_sides=pad_all_sides, memmap_mode='r')
    return dl_tr, dl_val, data_aug_params


def get_moreDA_augmentation(dataloader_train, dataloader_val, patch_size, params=default_3D_augmentation_params
                            , border_val_seg=-1, seeds_train=None, seeds_val=None, order_seg=1, order_data=3
                            , deep_supervision_scales=None, soft_ds=False, classes=None, pin_memory=True, regions=None,
                            use_nondetMultiThreadedAugmenter: bool = False):
    assert params.get('mirror') is None, "old version of params, use new keyword do_mirror"

    tr_transforms = []

    if params.get("selected_data_channels") is not None:
        tr_transforms.append(DataChannelSelectionTransform(params.get("selected_data_channels")))

    if params.get("selected_seg_channels") is not None:
        tr_transforms.append(SegChannelSelectionTransform(params.get("selected_seg_channels")))

    # don't do color augmentations while in 2d mode with 3d data because the color channel is overloaded!!
    if params.get("dummy_2D") is not None and params.get("dummy_2D"):
        ignore_axes = (0,)
        tr_transforms.append(Convert3DTo2DTransform())
    else:
        ignore_axes = None

    tr_transforms.append(SpatialTransform(
        patch_size, patch_center_dist_from_border=None,
#        do_elastic_deform=params.get("do_elastic"), alpha=params.get("elastic_deform_alpha"),
#        sigma=params.get("elastic_deform_sigma"),
        do_rotation=params.get("do_rotation"), angle_x=params.get("rotation_x"), angle_y=params.get("rotation_y"),
        angle_z=params.get("rotation_z"), p_rot_per_axis=params.get("rotation_p_per_axis"),
        do_scale=params.get("do_scaling"), scale=params.get("scale_range"),
        border_mode_data=params.get("border_mode_data"), border_cval_data=0, order_data=order_data,
        border_mode_seg="constant", border_cval_seg=border_val_seg,
        order_seg=order_seg, random_crop=params.get("random_crop"), p_el_per_sample=params.get("p_eldef"),
        p_scale_per_sample=params.get("p_scale"), p_rot_per_sample=params.get("p_rot"),
        independent_scale_for_each_axis=params.get("independent_scale_factor_for_each_axis")
    ))

    if params.get("dummy_2D"):
        tr_transforms.append(Convert2DTo3DTransform())

    # we need to put the color augmentations after the dummy 2d part (if applicable). Otherwise the overloaded color
    # channel gets in the way
#    tr_transforms.append(GaussianNoiseTransform(p_per_sample=0.1))
#    tr_transforms.append(GaussianBlurTransform((0.5, 1.), different_sigma_per_channel=True, p_per_sample=0.2,
#                                               p_per_channel=0.5))
#    tr_transforms.append(BrightnessMultiplicativeTransform(multiplier_range=(0.75, 1.25), p_per_sample=0.15))

#    if params.get("do_additive_brightness"):
#        tr_transforms.append(BrightnessTransform(params.get("additive_brightness_mu"),
#                                                 params.get("additive_brightness_sigma"),
#                                                 True, p_per_sample=params.get("additive_brightness_p_per_sample"),
#                                                 p_per_channel=params.get("additive_brightness_p_per_channel")))

#    tr_transforms.append(ContrastAugmentationTransform(p_per_sample=0.15))
#    tr_transforms.append(SimulateLowResolutionTransform(zoom_range=(0.5, 1), per_channel=True,
#                                                        p_per_channel=0.5,
#                                                        order_downsample=0, order_upsample=3, p_per_sample=0.25,
#                                                        ignore_axes=ignore_axes))
#    tr_transforms.append(
#        GammaTransform(params.get("gamma_range"), True, True, retain_stats=params.get("gamma_retain_stats"),
#                       p_per_sample=0.1))  # inverted gamma

#    if params.get("do_gamma"):
#        tr_transforms.append(
#            GammaTransform(params.get("gamma_range"), False, True, retain_stats=params.get("gamma_retain_stats"),
#                           p_per_sample=params["p_gamma"]))

    if params.get("do_mirror") or params.get("mirror"):
        tr_transforms.append(MirrorTransform(params.get("mirror_axes")))

    if params.get("mask_was_used_for_normalization") is not None:
        mask_was_used_for_normalization = params.get("mask_was_used_for_normalization")
        tr_transforms.append(MaskTransform(mask_was_used_for_normalization, mask_idx_in_seg=0, set_outside_to=0))

    tr_transforms.append(RemoveLabelTransform(-1, 0))

    if params.get("move_last_seg_chanel_to_data") is not None and params.get("move_last_seg_chanel_to_data"):
        tr_transforms.append(MoveSegAsOneHotToData(1, params.get("all_segmentation_labels"), 'seg', 'data'))
        if params.get("cascade_do_cascade_augmentations") is not None and params.get(
                "cascade_do_cascade_augmentations"):
            if params.get("cascade_random_binary_transform_p") > 0:
                tr_transforms.append(ApplyRandomBinaryOperatorTransform(
                    channel_idx=list(range(-len(params.get("all_segmentation_labels")), 0)),
                    p_per_sample=params.get("cascade_random_binary_transform_p"),
                    key="data",
                    strel_size=params.get("cascade_random_binary_transform_size"),
                    p_per_label=params.get("cascade_random_binary_transform_p_per_label")))
            if params.get("cascade_remove_conn_comp_p") > 0:
                tr_transforms.append(
                    RemoveRandomConnectedComponentFromOneHotEncodingTransform(
                        channel_idx=list(range(-len(params.get("all_segmentation_labels")), 0)),
                        key="data",
                        p_per_sample=params.get("cascade_remove_conn_comp_p"),
                        fill_with_other_class_p=params.get("cascade_remove_conn_comp_max_size_percent_threshold"),
                        dont_do_if_covers_more_than_X_percent=params.get(
                            "cascade_remove_conn_comp_fill_with_other_class_p")))

    tr_transforms.append(RenameTransform('seg', 'target', True))

    if regions is not None:
        tr_transforms.append(ConvertSegmentationToRegionsTransform(regions, 'target', 'target'))

    if deep_supervision_scales is not None:
        if soft_ds:
            assert classes is not None
            tr_transforms.append(DownsampleSegForDSTransform3(deep_supervision_scales, 'target', 'target', classes))
        else:
            tr_transforms.append(DownsampleSegForDSTransform2(deep_supervision_scales, 0, 0, input_key='target',
                                                              output_key='target'))

    tr_transforms.append(NumpyToTensor(['data', 'target'], 'float'))
    tr_transforms = Compose(tr_transforms)

    # MS: To avoid problem that occurs when multi threded augmenter is used.
    batchgenerator_train = SingleThreadedAugmenter(dataloader_train, tr_transforms)

    val_transforms = [RemoveLabelTransform(-1, 0)]
    if params.get("selected_data_channels") is not None:
        val_transforms.append(DataChannelSelectionTransform(params.get("selected_data_channels")))
    if params.get("selected_seg_channels") is not None:
        val_transforms.append(SegChannelSelectionTransform(params.get("selected_seg_channels")))

    if params.get("move_last_seg_chanel_to_data") is not None and params.get("move_last_seg_chanel_to_data"):
        val_transforms.append(MoveSegAsOneHotToData(1, params.get("all_segmentation_labels"), 'seg', 'data'))

    val_transforms.append(RenameTransform('seg', 'target', True))

    if regions is not None:
        val_transforms.append(ConvertSegmentationToRegionsTransform(regions, 'target', 'target'))

    if deep_supervision_scales is not None:
        if soft_ds:
            assert classes is not None
            val_transforms.append(DownsampleSegForDSTransform3(deep_supervision_scales, 'target', 'target', classes))
        else:
            val_transforms.append(DownsampleSegForDSTransform2(deep_supervision_scales, 0, 0, input_key='target',
                                                               output_key='target'))

    val_transforms.append(NumpyToTensor(['data', 'target'], 'float'))
    val_transforms = Compose(val_transforms)

    # MS: To avoid problem that occurs when multi threded augmenter is used.
    batchgenerator_val = SingleThreadedAugmenter(dataloader_val, val_transforms)

    return batchgenerator_train, batchgenerator_val


def prepare_generators(config_file, deep_supervision_scales = None):

    config_data = load_config_file(config_file)
    dl_tr, dl_val, data_aug_params = get_basic_generators(config_data)

    pin_memory = True

    tr_gen, val_gen = get_moreDA_augmentation(
                dl_tr, dl_val,
                data_aug_params['patch_size_for_spatialtransform'],
                data_aug_params,
                deep_supervision_scales=deep_supervision_scales,
                pin_memory=pin_memory,
                use_nondetMultiThreadedAugmenter=False
            )

    #_, val_gen = get_moreDA_augmentation(
    #            dl_tr, dl_val,
    #            data_aug_params['patch_size_for_spatialtransform'],
    #            data_aug_params,
    #            deep_supervision_scales=deep_supervision_scales, 
    #            pin_memory=pin_memory, 
    #            use_nondetMultiThreadedAugmenter=False
    #        )

    #_, tr_gen = get_moreDA_augmentation(
    #            dl_val, dl_tr,
    #            data_aug_params['patch_size_for_spatialtransform'],
    #            data_aug_params,
    #            deep_supervision_scales=deep_supervision_scales,
    #            pin_memory=pin_memory,
    #            use_nondetMultiThreadedAugmenter=False
    #        )

    return tr_gen, val_gen


if __name__ == '__main__':
    config_data = load_config_file('config_gen.json')
    dl_tr, dl_val, data_aug_params = get_basic_generators(config_data)

    pin_memory = True
    deep_supervision_scales = None

    tr_gen, val_gen = get_moreDA_augmentation(
                dl_tr, dl_val,
                data_aug_params['patch_size_for_spatialtransform'],
                data_aug_params,
                deep_supervision_scales=deep_supervision_scales,
                pin_memory=pin_memory,
                use_nondetMultiThreadedAugmenter=False
            )

