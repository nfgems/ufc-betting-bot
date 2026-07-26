from pathlib import Path

from bs4 import BeautifulSoup


DASHBOARD_TEMPLATE = (
    Path(__file__).resolve().parents[1] / "src" / "web" / "templates" / "dashboard.html"
)


def test_fight_decisions_sits_between_limit_orders_and_issues_log():
    dashboard = DASHBOARD_TEMPLATE.read_text(encoding="utf-8")
    soup = BeautifulSoup(dashboard, "html.parser")
    limit_orders = soup.select_one("#limitOrderCards")
    fight_decisions_title = soup.select_one('[data-section="fight-decisions"]')
    fight_decisions_body = soup.select_one('[data-collapse="fight-decisions"]')
    issues_log = soup.select_one("#issuesFeed")

    assert limit_orders is not None
    assert fight_decisions_title is not None
    assert fight_decisions_body is not None
    assert issues_log is not None
    assert limit_orders.find_next_sibling() is fight_decisions_title
    assert fight_decisions_title.find_next_sibling() is fight_decisions_body

    issues_title = fight_decisions_body.find_next_sibling()
    assert issues_title is not None
    assert issues_title.get_text(" ", strip=True).startswith("Issues Log")
    assert issues_title.find_next_sibling() is issues_log


def test_portfolio_allocation_display_is_removed():
    dashboard = DASHBOARD_TEMPLATE.read_text(encoding="utf-8")
    soup = BeautifulSoup(dashboard, "html.parser")

    assert "Portfolio Allocation" not in soup.get_text(" ", strip=True)
    assert soup.select_one("#allocationChart") is None
    assert "renderAllocationChart" not in dashboard
