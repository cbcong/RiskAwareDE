"""The new experiment uses one locked runtime; history is a comparison only."""
from pathlib import Path
import pytest
from reproducibility.cohorts import read, equivalent, scientific_output, checksum
from reproducibility.runtimes import fixed_runtime_plan, RuntimePool
from reproducibility.fixed_environment import check, identity

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "data/reference/e"
JOBS = read(REFERENCE / "jobs.json")
METHODS = sorted({j["method"] for j in JOBS})

@pytest.fixture(scope="module")
def native(tmp_path_factory):
    plan, env = fixed_runtime_plan(JOBS)
    with RuntimePool(plan, env, REFERENCE / "protocol.json", tmp_path_factory.mktemp("fixed")) as pool:
        yield plan, env, pool

def test_all_jobs_use_one_isolated_environment(native):
    plan, env, _ = native
    assert plan["job_counts"] == {"fixed": 9624}
    assert set(plan["jobs"].values()) == {"fixed"}
    assert env["fixed"]["python_version"] == [3,12,4]
    assert check()["isolated_site_packages"]
    assert len(check()["packages"]) == 30

def test_former_mismatch_is_a_valid_new_result(native):
    _, _, pool = native
    job = next(j for j in JOBS if j["id"] == "r20260923_m06_nominal_search_s0014")
    first, second = pool.execute(job), pool.execute(job)
    old = read(REFERENCE / "results" / (job["id"]+".json"))
    assert equivalent(scientific_output(first),scientific_output(second))
    assert not equivalent(scientific_output(first),scientific_output(old))
    assert first["environment_lock_sha256"] == identity()

@pytest.mark.parametrize("method", METHODS)
def test_each_method_is_repeatable_in_the_fixed_runtime(native, method):
    _, env, pool = native
    candidates = [j for j in JOBS if j["method"] == method]
    job = candidates[len(candidates)//2]
    first, second = pool.execute(job), pool.execute(job)
    assert equivalent(scientific_output(first),scientific_output(second)),job["id"]
    assert first["detail"].get("objective_history") == second["detail"].get("objective_history")
    assert first["checkpoint_sha256"] == checksum(first)
    assert first["reproduction_runtime"] == env["fixed"]

def test_changed_dependency_lock_is_rejected(monkeypatch, tmp_path):
    from reproducibility import fixed_environment as module
    invalid=tmp_path/"environment.lock.json"
    invalid.write_text("{}")
    monkeypatch.setattr(module,"LOCK",invalid)
    with pytest.raises(RuntimeError,match="differs from"):
        module.check()
