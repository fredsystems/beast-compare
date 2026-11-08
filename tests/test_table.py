import polars as pl
from frames import Row, arrival, frames

from beast_compare.cluster import cluster
from beast_compare.table import add_visits, build_table, heard, signal, valid
from beast_compare.validity import MINUTE_NS

RECEIVERS = ["a", "b"]
US = 12


def test_build_table() -> None:
    rows = [
        Row("a", arrival(1.0), 0, 1, signal=10, df=17),
        Row("b", arrival(1.1), 0, 1, signal=20, df=17),
        Row("a", arrival(70.0), 0, 2, signal=30, df=16),
    ]
    data = frames(rows, RECEIVERS).with_columns(
        t_ref=pl.Series([1e6 * US, 1e6 * US, 70e6 * US]),
        ref_segment=pl.Series([0, 0, 0], dtype=pl.UInt32),
    )
    clustered = cluster(data)
    first_minute = arrival(0) // MINUTE_NS * MINUTE_NS
    validity = pl.DataFrame(
        {
            "receiver": ["a", "a", "b", "b"],
            "minute_ns": [first_minute, first_minute + MINUTE_NS] * 2,
            "valid": [True, True, True, False],
        },
        schema_overrides={"receiver": pl.Enum(RECEIVERS)},
    )
    attributed = clustered.transmissions.select("tx").with_columns(range_nm=pl.lit(5.0))
    positions = clustered.transmissions.select("tx").head(1)
    table = build_table(clustered, attributed, positions, validity, RECEIVERS)

    assert table[heard("a")].to_list() == [True, True]
    assert table[heard("b")].to_list() == [True, False]
    assert table[signal("b")].to_list() == [20, None]
    assert table[valid("b")].to_list() == [True, False]
    assert table["wall_s"][0] == arrival(1.0) / 1e9
    assert table["df_group"].to_list() == [3, 0]
    assert table["position"].to_list() == [True, False]
    assert table["density"].to_list() == [1, 1]


def test_add_visits_splits_on_long_gaps() -> None:
    table = pl.DataFrame(
        {
            "address": [1, 1, 2, 1, 1, None],
            "wall_s": [0.0, 100.0, 50.0, 450.0, 460.0, 10.0],
        },
        schema_overrides={"address": pl.UInt32},
    )
    visits = add_visits(table, visit_gap_s=300)["visit"].to_list()
    # Address 1: 0 and 100 together; 450 and 460 after a 350 s gap.
    assert visits[0] == visits[1]
    assert visits[3] == visits[4] != visits[0]
    assert visits[2] not in (visits[0], visits[3])
    # No address: a visit of its own.
    assert visits[5] not in (visits[0], visits[2], visits[3])
