
class ModelConfig:

    __slots__ = [
        'model',
        'model_args',
        'optimizer',
        'weight_init',
        'lr_equal',
        'spectr_norm',
    ]

    def __init__(
        self,
        model,
        optimizer        = None,
        model_args       = None,
        weight_init      = None,
        lr_equal         = False,
        spectr_norm      = False,
    ):
        # pylint: disable=too-many-arguments
        self.model      = model
        self.model_args = model_args or {}
        self.optimizer  = optimizer or {
            'name' : 'AdamW', 'betas' : (0.5, 0.999), 'weight_decay' : 1e-5,
        }
        self.weight_init = weight_init
        self.lr_equal    = lr_equal
        self.spectr_norm = spectr_norm

    def to_dict(self):
        return { x : getattr(self, x) for x in self.__slots__ }

