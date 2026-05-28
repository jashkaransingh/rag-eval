from rageval.datasets import (generate_from_chunks, load_testset,
                              save_testset)
from rageval.judge_llm import StubJudge
from rageval.types import TestCase


def test_save_and_load_roundtrip(tmp_path):
    cases = [
        TestCase(question="q1", relevant_ids=["a:0"],
                 reference_answer="ans1", metadata={"cat": "x"}),
        TestCase(question="q2", relevant_ids=["b:1", "b:2"]),
    ]
    path = tmp_path / "ts.json"
    save_testset(cases, str(path))
    loaded = load_testset(str(path))

    assert len(loaded) == 2
    assert loaded[0].question == "q1"
    assert loaded[0].relevant_ids == ["a:0"]
    assert loaded[0].reference_answer == "ans1"
    assert loaded[0].metadata == {"cat": "x"}
    assert loaded[1].relevant_ids == ["b:1", "b:2"]


def test_generate_from_chunks_with_stub():
    chunks = [
        {"chunk_id": "doc:0", "source": "doc.md",
         "text": "Acme Robotics had Q3 revenue of 42 million. Growth came from automotive."},
        {"chunk_id": "doc:1", "source": "doc.md",
         "text": "The Mark VII weighs 142 kilograms with a 120 kg payload."},
    ]
    cases = generate_from_chunks(chunks, judge=StubJudge(),
                                 questions_per_chunk=1)
    assert len(cases) == 2
    # The stub points relevant_ids back at the source chunk
    assert cases[0].relevant_ids == ["doc:0"]
    assert cases[1].relevant_ids == ["doc:1"]
    for c in cases:
        assert c.question
        assert c.metadata.get("synthetic") == "stub"


def test_load_example_testset():
    cases = load_testset("examples/testset.json")
    assert len(cases) == 6
    for c in cases:
        assert c.question
        assert c.relevant_ids
        assert c.reference_answer
