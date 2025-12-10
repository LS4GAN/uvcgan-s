from uvcgan_s.base.weight_init import init_weights
from uvcgan_s.torch.funcs      import prepare_model
from uvcgan_s.torch.lr_equal   import apply_lr_equal
from uvcgan_s.torch.spectr_norm import apply_sn

def default_model_init(model, model_config, device):
    model = prepare_model(model, device)
    init_weights(model, model_config.weight_init)

    if model_config.lr_equal:
        apply_lr_equal(model)

    if model_config.spectr_norm:
        apply_sn(model)

    return model

