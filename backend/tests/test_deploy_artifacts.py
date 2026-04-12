import os
import tempfile

from core.deploy_artifacts import sync_inference_category_for_deploy


def test_sync_writes_env_and_ci_file():
    tmp = tempfile.mkdtemp()
    written = sync_inference_category_for_deploy(tmp, "timesformer")
    assert len(written) == 2
    env_p = os.path.join(tmp, "deploy", "docker-inference.env")
    ci_p = os.path.join(tmp, "deploy", "ci_inference_category")
    assert env_p in written and ci_p in written
    with open(env_p, encoding="utf-8") as f:
        body = f.read()
    assert "ISOCOURT_INFERENCE_CATEGORY=timesformer" in body
    with open(ci_p, encoding="utf-8") as f:
        assert f.read().strip() == "timesformer"
