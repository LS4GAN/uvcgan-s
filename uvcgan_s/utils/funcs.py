import copy

import torch
from torchvision.datasets.folder import default_loader

def recursive_update_dict(base_dict, new_dict):
    if new_dict is None:
        return

    for k,v in new_dict.items():
        if (
                isinstance(v, dict)
            and k in base_dict
            and isinstance(base_dict[k], dict)
        ):
            recursive_update_dict(base_dict[k], v)
        else:
            base_dict[k] = copy.deepcopy(v)

def join_dicts(*dicts_list):
    base_dict = {}

    for d in dicts_list:
        recursive_update_dict(base_dict, d)

    return base_dict

def check_value_in_range(value, value_range, hint = None):
    if value in value_range:
        return

    msg = ''

    if hint is not None:
        msg = hint + ' '

    msg += f"value '{value}' is not range {value_range}"

    raise ValueError(msg)

@torch.no_grad()
def load_image_fuzzy(path, transforms, device):
    result = None

    try:
        result = default_loader(path)
    except FileNotFoundError:
        base_path = path.split('.', maxsplit = 1)[0]

        for ext in [ '.png', '.jpg', '.jpg.png' ]:
            try:
                result = default_loader(base_path + ext)
            except FileNotFoundError:
                continue

    result = transforms(result)
    return result.to(device).unsqueeze(0)

