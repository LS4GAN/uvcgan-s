
class GradientCacher:

    def __init__(self, model, loss, cache_period = 10):
        self._cache_period = cache_period
        self._cache_grad  = {}
        self._cache_loss  = None
        self._iter        = 0
        self._model       = model
        self._loss        = loss

    def __call__(self, *args, **kwargs):
        if (
                (self._iter < self._cache_period)
            and (self._cache_loss is not None)
        ):
            for name, p in self._model.named_parameters():
                cached_grad = self._cache_grad[name]

                if cached_grad is None:
                    p.grad = None
                else:
                    p.grad = cached_grad.clone()

            result = self._cache_loss
            self._iter += 1

        else:
            result = self._loss(*args, **kwargs)
            result.backward()

            self._cache_loss = result
            self._cache_grad = {
                name : p.grad.detach().clone() if p.grad is not None else None
                    for (name, p) in self._model.named_parameters()
            }

            self._iter = 0

        return result

