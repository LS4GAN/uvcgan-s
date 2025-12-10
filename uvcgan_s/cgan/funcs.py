import torch

def set_two_domain_input(images, inputs, domain, device):
    if (domain is None) or (domain == 'both'):
        images.real_a = inputs[0].to(device, non_blocking = True)
        images.real_b = inputs[1].to(device, non_blocking = True)

    elif domain in [ 'a', 0 ]:
        images.real_a = inputs.to(device, non_blocking = True)

    elif domain in [ 'b', 1 ]:
        images.real_b = inputs.to(device, non_blocking = True)

    else:
        raise ValueError(
            f"Unknown domain: '{domain}'."
            " Supported domains: 'a' (alias 0), 'b' (alias 1), or 'both'"
        )

def set_asym_two_domain_input(images, inputs, domain, device):
    if (domain is None) or (domain == 'all'):
        images.real_a0 = inputs[0].to(device, non_blocking = True)
        images.real_a1 = inputs[1].to(device, non_blocking = True)
        images.real_b  = inputs[2].to(device, non_blocking = True)

    elif domain in [ 'a0', 0 ]:
        images.real_a0 = inputs.to(device, non_blocking = True)

    elif domain in [ 'a1', 1 ]:
        images.real_a1 = inputs.to(device, non_blocking = True)

    elif domain in [ 'b', 2 ]:
        images.real_b = inputs.to(device, non_blocking = True)

    else:
        raise ValueError(
            f"Unknown domain: '{domain}'."
            " Supported domains: 'a0' (alias 0), 'a1' (alias 1), "
            "'b' (alias 2), or 'all'"
        )

def trace_models(models, input_shapes, device):
    result = {}

    for (name, model) in models.items():
        if name not in input_shapes:
            continue

        shape = input_shapes[name]
        data  = torch.randn((1, *shape)).to(device)

        result[name] = torch.jit.trace(model, data)

    return result

