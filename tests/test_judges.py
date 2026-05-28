from rageval.judge_llm import StubJudge, _parse_json_verdict, build_judge
from rageval.metrics import judges as J
from rageval.types import RAGOutput, RetrievedChunk, TestCase


def test_parse_clean_json():
    v = _parse_json_verdict('{"score": 0.8, "rationale": "good"}')
    assert v.score == 0.8
    assert v.rationale == "good"


def test_parse_json_in_code_fence():
    text = '```json\n{"score": 0.5, "rationale": "ok"}\n```'
    v = _parse_json_verdict(text)
    assert v.score == 0.5


def test_parse_json_with_prose():
    text = 'Here is my verdict: {"score": 0.9, "rationale": "great"} done'
    v = _parse_json_verdict(text)
    assert v.score == 0.9


def test_parse_clamps_out_of_range():
    v = _parse_json_verdict('{"score": 1.5, "rationale": "x"}')
    assert v.score == 1.0
    v = _parse_json_verdict('{"score": -0.3, "rationale": "x"}')
    assert v.score == 0.0


def test_parse_bare_number_fallback():
    v = _parse_json_verdict("the score is 0.7")
    assert v.score == 0.7


def test_build_judge_falls_back_to_stub_without_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    judge = build_judge("anthropic")
    assert judge.name == "stub"


def test_stub_judge_faithfulness_rewards_overlap():
    judge = StubJudge()
    case = TestCase(question="What is the payload?")
    output = RAGOutput(
        answer="The payload capacity is 120 kilograms continuous.",
        retrieved=[RetrievedChunk(
            chunk_id="specs:0",
            text="The Mark VII payload capacity is 120 kilograms continuous, 180 peak.",
            score=0.9)])
    v = J.faithfulness(judge, case, output)
    assert v.score > 0.5


def test_stub_judge_context_precision_handles_no_answer():
    judge = StubJudge()
    case = TestCase(question="What is the battery life of the Mark VII?")
    output = RAGOutput(
        answer="irrelevant",
        retrieved=[RetrievedChunk(
            chunk_id="specs:1",
            text="Battery life of the Mark VII is 6 hours typical, 4 hours peak.",
            score=0.8)])
    v = J.context_precision(judge, case, output)
    assert v.score > 0.0  # the bug we fixed, used to return 0


def test_answer_correctness_none_without_reference():
    judge = StubJudge()
    case = TestCase(question="x", reference_answer=None)
    output = RAGOutput(answer="y")
    assert J.answer_correctness(judge, case, output) is None


def test_answer_correctness_with_reference():
    judge = StubJudge()
    case = TestCase(question="What is the capital of France?",
                    reference_answer="The capital of France is Paris.")
    output = RAGOutput(answer="The capital of France is Paris.")
    v = J.answer_correctness(judge, case, output)
    assert v is not None
    assert v.score > 0.5
