import pytest

from src.evaluation import box_iou, evaluate_at, image_level_metrics, match_boxes


def test_iou_identical_and_disjoint():
    assert box_iou([0, 0, 10, 10], [0, 0, 10, 10]) == pytest.approx(1.0)
    assert box_iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0


def test_iou_partial():
    assert box_iou([0, 0, 10, 10], [5, 0, 15, 10]) == pytest.approx(1 / 3)


def test_match_boxes():
    tp, fp, fn, ious = match_boxes([[0, 0, 10, 10]], [[0, 0, 10, 10], [50, 50, 60, 60]], [0.9, 0.8])
    assert (tp, fp, fn) == (1, 1, 0) and ious[0] == pytest.approx(1.0)


def test_image_level_metrics():
    m = image_level_metrics(tp=8, fp=2, tn=18, fn=2)
    assert m["sensitivity"] == pytest.approx(0.8) and m["specificity"] == pytest.approx(0.9)
    assert m["precision"] == pytest.approx(0.8)


def test_undefined_specificity_is_none():
    assert image_level_metrics(tp=1, fp=0, tn=0, fn=0)["specificity"] is None


def test_evaluate_at_counts():
    preds = [{"path": "a", "boxes": [[0, 0, 10, 10]], "scores": [0.9]},
             {"path": "b", "boxes": [[0, 0, 10, 10]], "scores": [0.2]},
             {"path": "c", "boxes": [], "scores": []}]
    gts = {"a": [[0, 0, 10, 10]], "b": [], "c": []}
    r = evaluate_at(preds, gts, conf=0.5)["image"]
    assert (r["tp"], r["fp"], r["tn"], r["fn"]) == (1, 0, 2, 0)
