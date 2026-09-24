import unittest

import torch

from utils.optimizers import build_scheduler


class SchedulerTests(unittest.TestCase):
    def test_linear_warmup_hands_off_to_cosine(self):
        first = torch.nn.Parameter(torch.tensor(1.0))
        second = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.SGD([{"params": [first], "lr": 1.0},
                                     {"params": [second], "lr": 0.1}])
        scheduler = build_scheduler(optimizer, "cosine", total_step=6, warmup_step=2,
                                    warmup_start_factor=0.1)
        observed = [[group["lr"] for group in optimizer.param_groups]]
        for _ in range(6):
            optimizer.step()
            scheduler.step()
            observed.append([group["lr"] for group in optimizer.param_groups])

        expected = [0.1, 0.55, 1.0, 0.5 * (1 + 2 ** -0.5), 0.5, 0.5 * (1 - 2 ** -0.5), 0.0]
        for actual, multiplier in zip(observed, expected):
            self.assertAlmostEqual(actual[0], multiplier, places=6)
            self.assertAlmostEqual(actual[1], 0.1 * multiplier, places=6)

    def test_invalid_warmup_settings(self):
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        for warmup_step, start_factor in ((-1, .1), (10, .1), (1, 0.), (1, 1.1)):
            optimizer = torch.optim.SGD([parameter], lr=1.0)
            with self.assertRaises(ValueError):
                build_scheduler(optimizer, "cosine", total_step=10, warmup_step=warmup_step,
                                warmup_start_factor=start_factor)


if __name__ == "__main__":
    unittest.main()
