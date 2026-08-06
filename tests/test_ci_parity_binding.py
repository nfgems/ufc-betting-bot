from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_generic_ci_builds_and_binds_one_exact_parity_candidate():
    workflow = (REPO_ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")

    build_step = workflow.index("- name: Build exact parity candidate")
    test_step = workflow.index("- name: Run tests")
    assert build_step < test_step
    assert "scripts/rebuild_ufc_processed_artifacts.py" in workflow
    assert 'test -s "$parity_dir/fights_cleaned.csv"' in workflow
    assert 'test -s "$parity_dir/features.csv"' in workflow
    assert "UFC_PARITY_PROCESSED_DATA_DIR=$parity_dir" in workflow
    assert (
        "UFC_PARITY_TRAINING_SPEC="
        "full_live_contract_v6_durability_corrected_20260805_fullfit"
    ) in workflow
