import pytest
from src.output_parser import parse_model_answer


@pytest.mark.parametrize(
    ("raw_output", "expected"),
    [
        ("Paris", "Paris"),
        ("Final answer: Paris", "Paris"),
        ("Answer:\nParis\nExplanation: ...", "Paris"),
        ('{"answer": "Paris"}', "Paris"),
        ({"answer": "Paris"}, "Paris"),
    ],
)
def test_valid_and_recoverable_outputs(raw_output, expected):
    assert parse_model_answer(raw_output) == expected


@pytest.mark.parametrize(
    "raw_output",
    [
        None,
        123,
        "",
        "   \n ",
        '{"answer":',
        {"answer": 123},
    ],
)
def test_malformed_outputs_are_handled_safely(raw_output):
    assert parse_model_answer(raw_output) == ""