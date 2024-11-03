import torch


def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Union[Tensor, List]]): dict, containing batch-version
            of the tensors or lists.
    """
    pack_to_tensors_batch_keys = [
        "mix_spectrogram",
        "mix_magnitude",
        "mix_phase",
        "s1_spectrogram",
        "s2_spectrogram",
        "s1_video",
        "s2_video",
        "mix",
        "s1",
        "s2",
    ]
    pack_to_list_batch_keys = ["audio_path"]
    result_batch = {}
    for key in pack_to_tensors_batch_keys + pack_to_list_batch_keys:
        if dataset_items[0][key] is None:  # e.g video
            result_batch[key] = None
            continue

        list_of_batch_values = []
        for item in dataset_items:
            list_of_batch_values.append(item[key])

        if key in pack_to_tensors_batch_keys:
            result_batch[key] = torch.cat(list_of_batch_values, dim=0)
        else:
            result_batch[key] = list_of_batch_values
    return result_batch
