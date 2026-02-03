from app.services.langchain_matcher import _remove_input_from_candidates

def test_removes_exact_match():
    input_name = "DWP"
    candidates = ["DWP", "Department for Working Pensions", "HMRC"]

    out = _remove_input_from_candidates(input_name, candidates)

    assert out == ["Department for Working Pensions", "HMRC"]


def test_removes_all_duplicates():
    input_name = "DWP"
    candidates = ["DWP", "HMRC", "DWP", "DWP"]

    out = _remove_input_from_candidates(input_name, candidates)

    assert out == ["HMRC"]


def test_noop_when_not_present_and_does_not_mutate_original():
    input_name = "DWP"
    candidates = ["HMRC", "Department for Working Pensions"]
    original = list(candidates)

    out = _remove_input_from_candidates(input_name, candidates)

    assert candidates == original  # original list unchanged
    assert out == original  # output has same values
    assert out is not candidates  # but it's a new list
