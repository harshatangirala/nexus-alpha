import pandas as pd
import pytest

from nexus_alpha import splits as S
from nexus_alpha.data import news as N
from nexus_alpha.data.calendar import offset_session, sessions


def _all_sessions(a="2002-06-01", b="2022-12-30"):
    s = sessions()
    return pd.Series(s[(s >= pd.Timestamp(a)) & (s <= pd.Timestamp(b))])


def test_primary_blocks_are_purged_and_embargoed():
    asof = _all_sessions()
    blk = S.assign_blocks(asof)
    df = pd.DataFrame({"asof_date": asof.values, "block": blk.values})
    rep = S.verify_integrity(df)
    for k, v in rep.items():
        assert v["gap_sessions"] >= 0, k
    # the last train row: its 21-session label end plus a 5-session embargo is before the first inner session
    last_train = df[df.block == "train"].asof_date.max()
    end = offset_session(pd.DatetimeIndex([last_train]), 21 + 5)[0]
    assert end < df[df.block == "inner"].asof_date.min()
    # nothing before 2003, nothing assigned in the purged tail of a block
    assert df[df.asof_date < "2003-01-01"].block.isna().all()
    assert df.block.notna().sum() > 4500


def test_verify_integrity_catches_a_leaking_row():
    asof = _all_sessions()
    df = pd.DataFrame({"asof_date": asof.values, "block": S.assign_blocks(asof).values})
    bad = df[(df.asof_date > "2015-12-20") & (df.asof_date <= "2015-12-31")].index
    df.loc[bad, "block"] = "train"                      # put the purged rows back into train
    with pytest.raises(AssertionError):
        S.verify_integrity(df)


def test_walk_forward_folds_cover_crises():
    assert 2008 in S.WALK_FORWARD_TEST_YEARS and 2020 in S.WALK_FORWARD_TEST_YEARS and 2022 in S.WALK_FORWARD_TEST_YEARS
    asof = _all_sessions()
    blocks = S.walk_forward_blocks(2008)
    blk = S.assign_blocks(asof, blocks)
    df = pd.DataFrame({"asof_date": asof.values, "block": blk.values})
    S.verify_integrity(df, blocks=blocks)
    assert df[df.block == "test"].asof_date.dt.year.eq(2008).all()


# ------------------------------------------------------------------ news point-in-time policy
def test_minute_day_and_midnight_availability():
    dates = pd.Series(["2015-03-10T14:31:00Z", "2015-03-10T00:00:00Z", "2015-03-10T00:00:00Z"])
    prec = pd.Series(["minute", "minute", "day"])
    pub, avail, eff = N.availability(dates, prec)
    assert avail[0] == pd.Timestamp("2015-03-10 14:46", tz="UTC")
    assert eff[1] == "day_from_midnight" and avail[1] == pd.Timestamp("2015-03-11 08:00", tz="UTC")
    assert avail[2] == pd.Timestamp("2015-03-11 08:00", tz="UTC")


def test_article_after_decision_time_is_excluded():
    idx = pd.DataFrame({"ticker": ["AAPL"] * 4, "available_utc": pd.to_datetime(
        ["2020-06-01 12:00", "2020-06-02 12:59", "2020-06-02 13:01", "2020-05-01 12:00"], utc=True)})
    decision = pd.Timestamp("2020-06-02 09:00", tz="America/New_York").tz_convert("UTC")   # 13:00 UTC
    got = N.available_for(idx, "AAPL", decision, pd.Timedelta(days=5))
    assert got.available_utc.tolist() == list(pd.to_datetime(["2020-06-01 12:00", "2020-06-02 12:59"], utc=True))


def test_day_level_article_dated_asof_day_is_usable_next_morning_but_not_same_day():
    # a day-level article dated Tuesday 2020-06-02 (time unknown): not usable for Tuesday's 09:00 decision, usable Wednesday
    _, avail, _ = N.availability(pd.Series(["2020-06-02T00:00:00Z"]), pd.Series(["day"]))
    tue = pd.Timestamp("2020-06-02 09:00", tz="America/New_York").tz_convert("UTC")
    wed = pd.Timestamp("2020-06-03 09:00", tz="America/New_York").tz_convert("UTC")
    assert not (avail[0] <= tue) and avail[0] <= wed


def test_ticker_normalisation():
    assert N.normalize_ticker("$aapl") == "AAPL"
    assert N.normalize_ticker("BRK-B") == "BRK.B" and N.normalize_ticker("brk/b") == "BRK.B"
    assert N.normalize_ticker("not a ticker") is None and N.normalize_ticker("") is None
    assert N.tickers_of({"stocks": ["AAPL", "aapl", "MSFT"]}) == ["AAPL", "MSFT"]
    assert N.tickers_of({"stocks": "AAPL, MSFT"}) == ["AAPL", "MSFT"]


def test_near_duplicate_headlines_cluster_to_the_earliest_copy():
    heads = ["Apple beats estimates as iPhone sales jump",
             "Apple beats estimates as iPhone sales jump - update",
             "Microsoft announces new CEO",
             "Apple beats estimates as iPhone sales jump"]
    t = pd.to_datetime(["2020-01-02 13:00", "2020-01-02 15:00", "2020-01-02 16:00", "2020-01-09 13:00"], utc=True)
    cid = N.cluster_near_duplicates(heads, t, threshold=0.7)
    assert cid[0] == cid[1] == 0          # syndicated copy joins the earliest article's cluster
    assert cid[2] != cid[0]
    assert cid[3] != cid[0]               # outside the 2-day window: a new cluster, not merged across a week
    assert N.text_hash("Hello,  World!") == N.text_hash("hello world")
