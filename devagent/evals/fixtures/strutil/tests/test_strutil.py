from src.strutil import slugify, truncate


def test_slugify_simple():
    assert slugify("Hello World") == "hello-world"


def test_slugify_collapses_and_strips():
    assert slugify("  Hello,  World! ") == "hello-world"


def test_truncate():
    assert truncate("abcdefgh", 6) == "abc..."
    assert truncate("abc", 6) == "abc"
