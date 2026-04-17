from qlab.cost import CryptoCost, FxCost


class TestCryptoCost:

    def test_default_5bps(self):
        c = CryptoCost()
        assert c.round_trip_cost(10_000) == 10.0  # 5bps * 2 * $10k

    def test_custom_bps(self):
        c = CryptoCost(taker_bps=2.5)
        assert c.round_trip_cost(10_000) == 5.0


class TestFxCost:

    def test_spread_cost(self):
        c = FxCost(spread_pips=2.0, pip_value=0.0001)
        assert abs(c.round_trip_cost(100_000) - 20.0) < 1e-10

    def test_swap_cost(self):
        c = FxCost(swap_per_day=0.5)
        assert c.swap_cost(1.0, 10) == 5.0

    def test_total_cost(self):
        c = FxCost(spread_pips=1.0, pip_value=0.0001, swap_per_day=0.0001)
        total = c.total_cost(100_000, 5)
        spread = 100_000 * 1.0 * 0.0001   # 10
        swap = 100_000 * 0.0001 * 5        # 50
        assert abs(total - (spread + swap)) < 1e-10
