"""Integration checks for shared actual-vs-expected and newsletter entry points."""

from pathlib import Path
import unittest

from jinja2 import Environment, FileSystemLoader
from actual_vs_expected import build_actual_vs_expected

ROOT = Path(__file__).resolve().parents[1]
TEMPLATES = ROOT / "templates"


class NewsletterAndFplSectionTests(unittest.TestCase):
    def test_shared_chart_renders_with_realistic_season_record(self):
        views = build_actual_vs_expected([{
            "Team": "Arsenal", "MP": 8, "G": 15, "xG": 14,
            "GA": 8, "xGA": 9, "PTS": 18, "xPTS": 16,
        }])
        env = Environment(loader=FileSystemLoader(str(TEMPLATES)))
        rendered = env.get_template("_actual_vs_expected.html").render(actual_vs_expected=views)
        self.assertIn("Actual vs Expected", rendered)
        self.assertIn('data-view="scoring"', rendered)
        self.assertIn('data-view="defending"', rendered)
        self.assertIn('data-view="points"', rendered)

    def test_chart_hidden_when_no_results(self):
        env = Environment(loader=FileSystemLoader(str(TEMPLATES)))
        rendered = env.get_template("_actual_vs_expected.html").render(actual_vs_expected=None)
        self.assertNotIn("actualVsExpectedChart", rendered)

    def test_fpl_chart_before_fixture_difficulty(self):
        source = (TEMPLATES / "fpl.html").read_text()
        self.assertLess(
            source.index('{% include "_actual_vs_expected.html" %}'),
            source.index('Fixture Difficulty - Next 5 Gameweeks'),
        )
        self.assertIn('{% include "_actual_vs_expected_script.html" %}', source)

    def test_both_pages_reuse_newsletter_cta(self):
        for page in ("premier_league.html", "fpl.html"):
            source = (TEMPLATES / page).read_text()
            self.assertIn('{% include "_newsletter_cta.html" %}', source)
        self.assertIn('href="/#newsletter"', (TEMPLATES / "_newsletter_cta.html").read_text())

    def test_analytics_opt_out_precedes_ga_load(self):
        source = (TEMPLATES / "base.html").read_text()
        self.assertLess(
            source.index("fivestat_analytics_disabled"),
            source.index("googletagmanager.com/gtag/js"),
        )
        self.assertIn("window.fivestatAnalyticsEnabled", source)


if __name__ == "__main__":
    unittest.main()
