"""The design system: stylesheet renders, components escape input, icons and palette are consistent."""
import re

from core import charts, components as C, theme


def test_stylesheet_renders_and_is_balanced():
    css = theme.css()                                   # raises if a bare % breaks the template
    assert css.count("{") == css.count("}")
    assert "--ic-primary" in css and ".ic-ico-check" in css and "@media (max-width:640px)" in css
    assert not re.search(r"%\(", css)                    # no unformatted template tokens left


def test_every_icon_has_a_css_mask_and_components_use_known_icons():
    css = theme.css()
    for name in theme.ICONS:
        assert f".ic-ico-{name}" in css
    assert "ic-ico-check" in C.icon("check")


def test_components_escape_user_text():
    assert "<script>" not in C.chip_html("<script>alert(1)</script>")
    assert "&lt;b&gt;" in C.kv_html([("k", "<b>x</b>")])
    assert "<img" not in C.health_card_html("1", "<img src=x>", "help")


def test_stepper_states():
    html = C.stepper_html(2, {1})
    assert html.count("ic-step ") >= 5 and 'class="ic-step done"' in html and 'class="ic-step current"' in html
    assert C.stepper_html(0, set()).count("current") == 0
    assert C.initials("Anna Maria Auer") == "AA" and C.initials("") == "?"


def test_chart_palette_and_semantic_colours():
    assert charts.COLORS == theme.PALETTE and len(set(theme.PALETTE)) == len(theme.PALETTE)
    assert charts.GOOD != charts.BAD
    theme.register_plotly_template()
    import plotly.io as pio
    assert pio.templates.default == "ic"
