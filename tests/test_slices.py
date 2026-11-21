"""Query slices for error analysis: each tag is a simple, explainable rule."""
import pandas as pd

from shoprank.eval.slices import length_bucket, query_tags


def test_length_bucket_counts_words():
    assert length_bucket("ipad") == "1-2 words"
    assert length_bucket("ipad bay bag") == "3-5 words"
    assert length_bucket("womens black running shoes size 8 wide") == "6+ words"


def test_query_tags_find_numbers_negation_and_spanish():
    tags = query_tags("freeze dryer machine not dehydrator", brands=set())
    assert tags["negation"] and not tags["has_number"] and not tags["spanish_like"]
    assert query_tags("2.5 hdd", brands=set())["has_number"]
    assert query_tags("zapatos para mujer", brands=set())["spanish_like"]
    assert query_tags("niño juguete", brands=set())["spanish_like"]  # accented letter
    assert not query_tags("nothing bundt cakes", brands=set())["negation"]  # "nothing" is not "not"


def test_brand_tag_needs_a_whole_brand_word_from_the_querys_candidates():
    assert query_tags("turtle beach headset", brands={"turtle beach"})["brand"]
    assert not query_tags("turtles toy", brands={"turtle beach"})["brand"]
    assert not query_tags("hp ink", brands={"hp"})["brand"]  # brands under 3 letters are ignored


def test_tags_table_has_one_row_per_query():
    pairs = pd.DataFrame({"query_id": [1, 1, 2], "query": ["hp ink", "hp ink", "zapatos para mujer"],
                          "brand": ["HP", "Canon", "Nike"]})
    from shoprank.eval.slices import tag_queries
    table = tag_queries(pairs)
    assert table["query_id"].tolist() == [1, 2]
    assert table.loc[table.query_id == 2, "spanish_like"].item()
