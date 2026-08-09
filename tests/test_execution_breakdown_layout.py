from pathlib import Path

from bs4 import BeautifulSoup


EXECUTION_TEMPLATE = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "web"
    / "templates"
    / "execution_breakdown.html"
)


def _template():
    source = EXECUTION_TEMPLATE.read_text(encoding="utf-8")
    return source, BeautifulSoup(source, "html.parser")


def test_execution_page_has_labelled_controls_and_current_navigation():
    _, soup = _template()

    assert soup.select_one('nav[aria-label="Primary"]') is not None
    current = soup.select_one('a[aria-current="page"]')
    assert current is not None
    assert current.get_text(strip=True) == "Execution"
    assert soup.select_one('label[for="cycleSelect"] #cycleSelect') is not None
    assert soup.select_one('label[for="searchInput"] #searchInput') is not None
    assert soup.select_one('#pageMeta[aria-live="polite"]') is not None
    assert soup.select_one('#resultMeta[role="status"][aria-live="polite"]') is not None
    assert soup.select_one('#executionMain[aria-busy="true"]') is not None


def test_cycle_identity_is_separate_from_the_six_metric_summary():
    source, soup = _template()

    cycle_id = soup.select_one("#cycleId")
    assert cycle_id is not None
    assert cycle_id.find_parent(class_="cycle-hero") is not None
    assert cycle_id.find_parent(id="summaryGrid") is None

    for label in (
        "Fights",
        "Orders",
        "Already bet",
        "Blocked",
        "Skipped / waiting",
        "Errors",
    ):
        assert f"label:'{label}'" in source
    assert "strategy checks" in source


def test_decision_density_is_reduced_with_progressive_disclosure():
    source, soup = _template()

    assert soup.select_one("#filterBar") is not None
    assert "sharedFightBanner(fight)" in source
    assert "windows.length === pathKeys.length" in source
    assert "windowSignatures.size === 1" in source
    assert "Bet window closed" in source
    assert "Full audit trail" in source
    assert "Technical identifiers" in source
    assert "Decision stages" in source
    assert "grid-template-columns:repeat(2,minmax(0,1fr))" in source


def test_active_paths_exclude_g_but_legacy_cycles_render_it():
    source, _ = _template()

    assert "const ACTIVE_PATHS = ['S', 'C', 'M'];" in source
    assert "G:'Legacy G Trader (retired)'" in source
    assert "hasOwnProperty.call(fight?.paths || {}, 'G')" in source
    assert "hasLegacyG ? [...ACTIVE_PATHS, 'G'] : ACTIVE_PATHS" in source
    assert "visiblePaths().flatMap" in source
    assert "pathKeys.map(path => pathPanel" in source
