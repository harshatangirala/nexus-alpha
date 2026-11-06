"""v2 inference uses the same decision time as training and v1: 09:00 New York on the next session (D-003)."""
import pandas as pd

from nexus_alpha.inference.pipeline_v2 import decision_utc


def test_friday_maps_to_monday_0900_new_york_in_both_seasons():
    for d, utc_hour in (("2019-07-12", 13), ("2019-12-13", 14), ("2026-07-10", 13), ("2026-11-06", 14)):
        t = decision_utc(pd.Timestamp(d))
        assert t.day_name() == "Monday" and t.hour == utc_hour, (d, t)
        assert t.tz_convert("America/New_York").hour == 9
