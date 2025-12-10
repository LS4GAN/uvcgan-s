import unittest

import torch
from torch import nn

from uvcgan_s.torch.gradient_cacher import GradientCacher

SEED = 0
ALLCLOSE_ARGS = {
    'rtol'      : 1e-09,
    'atol'      : 1e-12,
    'equal_nan' : True,
}

class TestGradientCacher(unittest.TestCase):

    def _compare_gradients(self, model_1, model_2):
        it_1 = model_1.named_parameters()
        it_2 = model_2.named_parameters()

        for (n1, v1), (n2, v2) in zip(it_1, it_2):
            self.assertEqual(n1, n2)

            self.assertTrue(v1.grad is not None)

            self.assertTrue(torch.allclose(v1, v2, **ALLCLOSE_ARGS))
            self.assertTrue(torch.allclose(v1.grad, v2.grad, **ALLCLOSE_ARGS))

    def _verify_different_gradients(self, model_1, model_2):
        it_1 = model_1.named_parameters()
        it_2 = model_2.named_parameters()

        for (n1, v1), (n2, v2) in zip(it_1, it_2):
            self.assertEqual(n1, n2)
            self.assertTrue(v1.grad is not None)

            self.assertFalse(torch.allclose(v1.grad, v2.grad, **ALLCLOSE_ARGS))

    def _init_models(self, model_null, model_test):
        model_test.load_state_dict(model_null.state_dict())

        model_test.train()
        model_null.train()

    def _reinit_models(self, model_null, model_test):
        with torch.no_grad():
            for p in model_null.parameters():
                p.copy_(torch.randn(p.shape))

            model_test.load_state_dict(model_null.state_dict())

    def _init_optimizers(self, model_null, model_test, lr = 1):
        opt_null = torch.optim.SGD(model_null.parameters(), lr)
        opt_test = torch.optim.SGD(model_test.parameters(), lr)

        return opt_null, opt_test

    def test_simple_backward(self):
        torch.manual_seed(SEED)

        model_test = nn.Linear(10, 1)
        model_null = nn.Linear(10, 1)

        self._init_models(model_null, model_test)

        loss_fn     = nn.L1Loss()
        grad_cacher = GradientCacher(model_test, loss_fn, cache_period = 1)

        x      = torch.randn((4, 10))
        target = torch.randn((4, 1))

        y_test = model_test(x)
        y_null = model_null(x)

        loss_null = loss_fn(y_null, target)
        loss_null.backward()

        loss_test = grad_cacher(y_test, target)

        self._compare_gradients(model_test, model_null)

    def test_simple_cached_backward(self):
        cache_period = 4
        torch.manual_seed(SEED)

        # NOTE: bias gradient loops
        model_test = nn.Linear(10, 1, bias = False)
        model_null = nn.Linear(10, 1, bias = False)

        self._init_models(model_null, model_test)
        opt_null, opt_test = self._init_optimizers(model_null, model_test)

        loss_fn     = nn.L1Loss()
        grad_cacher = GradientCacher(model_test, loss_fn, cache_period)

        for i in range(10):
            model_test.load_state_dict(model_null.state_dict())

            opt_null.zero_grad(set_to_none = False)
            opt_test.zero_grad(set_to_none = False)

            x      = torch.randn((4, 10))
            target = torch.randn((4, 1))

            y_test = model_test(x)
            y_null = model_null(x)

            loss_null = loss_fn(y_null, target)
            loss_null.backward()

            loss_test = grad_cacher(y_test, target)

            if i % (cache_period+1) ==  0:
                self._compare_gradients(model_null, model_test)
            else:
                self._verify_different_gradients(model_null, model_test)

            opt_null.step()
            opt_test.step()

    def test_gradient_overwriting(self):
        cache_period = 1000
        torch.manual_seed(SEED)

        model_test = nn.Linear(10, 1)
        model_null = nn.Linear(10, 1)

        self._init_models(model_null, model_test)

        loss_fn     = nn.L1Loss()
        grad_cacher = GradientCacher(model_test, loss_fn, cache_period)

        x      = torch.randn((4, 10))
        target = torch.randn((4, 1))

        y_test = model_test(x)
        y_null = model_null(x)

        loss_null = loss_fn(y_null, target)
        loss_null.backward()

        # Cache
        loss_test = grad_cacher(y_test, target)

        for i in range(10):
            # Try to overwrite gradient tensor
            x      = torch.randn((4, 10))
            target = torch.randn((4, 1))
            y_test = model_test(x)

            loss_test_overwrite = loss_fn(y_test, target)
            loss_test_overwrite.backward()

        # Get from Cache
        loss_test = grad_cacher(y_test, target)
        self._compare_gradients(model_null, model_test)

if __name__ == '__main__':
    unittest.main()

