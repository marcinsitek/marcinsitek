import numpy as np
import pandas as pd
import random


class DataGenerator:
    def __init__(
        self,
        p_before=0.1,
        p_after_a=0.1,
        p_after_b=0.1,
        k=0.5,
        theta=2.0,
        mu=1.0,
        sigma=0.1,
    ):
        self.p_before = p_before
        self.p_after_a = p_after_a
        self.p_after_b = p_after_b
        self.k = k
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.C = 100

    def generate(self, N=10000, seed=89):
        np.random.seed(seed)
        variant = np.array(random.choices(population=["A", "B"], k=N))
        variant_a_count = (variant == "A").sum()
        payer_after_a = np.random.binomial(n=1, p=self.p_after_a, size=variant_a_count)
        payer_after_b = np.random.binomial(
            n=1, p=self.p_after_b, size=N - variant_a_count
        )
        payer_after = np.zeros(N)
        payer_after[variant == "A"] = payer_after_a
        payer_after[variant == "B"] = payer_after_b
        payer_after_count = (payer_after == 1).sum()
        payer_before = np.zeros(N)
        payer_before[payer_after == 1] = np.random.binomial(
            n=1, p=self.p_before, size=payer_after_count
        )
        payment_sum_before = (
            payer_before
            * np.random.gamma(shape=self.k, scale=self.theta, size=N)
            * self.C
        )
        payment_sum_a_after = (
            payer_after_a
            * np.random.gamma(shape=self.k, scale=self.theta, size=variant_a_count)
            * self.C
        )
        payment_sum_b_after = (
            payer_after_b
            * np.random.gamma(shape=self.k, scale=self.theta, size=N - variant_a_count)
            * self.mu
            * self.C
        )
        payment_sum_after = np.zeros(N)
        payment_sum_after[variant == "A"] = payment_sum_a_after
        payment_sum_after[variant == "B"] = payment_sum_b_after

        df = pd.DataFrame(
            {
                "player_id": range(1, N + 1),
                "variant": variant,
                "p_sum_before": payment_sum_before,
                "p_sum_after": payment_sum_after,
            }
        )
        return df
