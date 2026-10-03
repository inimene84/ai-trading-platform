"""The labeler must stay side-aware, finite-window, and must not invent labels."""

import unittest

import numpy as np
import pandas as pd

from triple_barrier import apply_triple_barrier, generate_meta_labels, get_atr, true_range


def _frame(rows: list[tuple]) -> pd.DataFrame:
    index = pd.date_range("2026-08-01", periods=len(rows), freq="1h", tz="UTC")
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=index)


def _quiet(n: int, close: float = 100.0) -> list[tuple]:
    # True range is 2 once a previous close exists: high-low is the widest leg.
    return [(close, close + 1.0, close - 1.0, close) for _ in range(n)]


class TripleBarrierTests(unittest.TestCase):
    def test_atr_is_the_mean_of_the_last_window_not_an_exponential_average(self):
        rows = _quiet(40)
        rows[1] = (100.0, 180.0, 100.0, 100.0)
        frame = _frame(rows)
        atr = get_atr(frame, period=14)
        self.assertTrue(np.isnan(atr.iloc[13]))
        # The spike at bar 1 is inside the first complete window and outside a later one.
        self.assertAlmostEqual(float(atr.iloc[14]), (80.0 + 13 * 2.0) / 14.0)
        self.assertAlmostEqual(float(atr.iloc[16]), 2.0)
        exponential = true_range(frame).ewm(span=14, adjust=False).mean()
        self.assertGreater(abs(float(atr.iloc[16]) - float(exponential.iloc[16])), 0.2)

    def test_long_stop_target_and_both_touch(self):
        rows = _quiet(20)
        # ATR at index 14 is 2. Stop 96.5, target 111.
        rows[15] = (100.0, 100.0, 96.5, 96.5)
        stopped = apply_triple_barrier(_frame(rows), events_idx=pd.Index([_frame(rows).index[14]]), max_holding_bars=4)
        self.assertEqual(int(stopped["label"].iloc[0]), -1)
        self.assertEqual(stopped["touch_type"].iloc[0], "sl")

        rows[15] = (100.0, 111.0, 99.0, 111.0)
        target = apply_triple_barrier(_frame(rows), events_idx=pd.Index([_frame(rows).index[14]]), max_holding_bars=4)
        self.assertEqual(int(target["label"].iloc[0]), 1)
        self.assertEqual(target["touch_type"].iloc[0], "pt")

        rows[15] = (100.0, 111.0, 96.5, 100.0)
        both = apply_triple_barrier(_frame(rows), events_idx=pd.Index([_frame(rows).index[14]]), max_holding_bars=4)
        self.assertEqual(both["touch_type"].iloc[0], "sl")
        self.assertEqual(int(both["label"].iloc[0]), -1)

    def test_timeout_stays_a_third_class_after_a_profitable_expiry(self):
        rows = _quiet(20)
        for cursor in range(15, 19):
            rows[cursor] = (104.0, 105.0, 99.0, 105.0)
        frame = _frame(rows)
        labeled = apply_triple_barrier(frame, events_idx=pd.Index([frame.index[14]]), max_holding_bars=4)
        self.assertEqual(labeled["touch_type"].iloc[0], "timeout")
        self.assertEqual(int(labeled["label"].iloc[0]), 0)
        self.assertGreater(float(labeled["ret"].iloc[0]), 0.0)

    def test_short_profit_is_a_down_move(self):
        rows = _quiet(20)
        rows[15] = (100.0, 101.0, 89.0, 89.0)
        frame = _frame(rows)
        sides = pd.Series(0, index=frame.index)
        sides.iloc[14] = -1
        labeled = apply_triple_barrier(
            frame,
            events_idx=pd.Index([frame.index[14]]),
            sides=sides,
            max_holding_bars=4,
        )
        self.assertEqual(int(labeled["label"].iloc[0]), 1)
        self.assertEqual(labeled["touch_type"].iloc[0], "pt")
        self.assertEqual(int(labeled["side"].iloc[0]), -1)

    def test_missing_warmup_missing_future_and_a_hole_are_omitted(self):
        rows = _quiet(20)
        frame = _frame(rows)
        early = apply_triple_barrier(frame, events_idx=pd.Index([frame.index[5]]), max_holding_bars=4)
        self.assertTrue(early.empty)
        tail = apply_triple_barrier(frame, events_idx=pd.Index([frame.index[18]]), max_holding_bars=4)
        self.assertTrue(tail.empty)
        gapped = frame.drop(frame.index[16])
        labeled = apply_triple_barrier(gapped, events_idx=pd.Index([gapped.index[14]]), max_holding_bars=4)
        self.assertTrue(labeled.empty)

    def test_meta_label_counts_a_timeout_as_failure(self):
        rows = _quiet(20)
        for cursor in range(15, 19):
            rows[cursor] = (104.0, 105.0, 99.0, 105.0)
        frame = _frame(rows)
        signals = pd.Series(0, index=frame.index)
        signals.iloc[14] = 1
        meta = generate_meta_labels(frame, signals, max_holding_bars=4)
        self.assertEqual(int(meta["meta_label"].iloc[0]), 0)


if __name__ == "__main__":
    unittest.main()
