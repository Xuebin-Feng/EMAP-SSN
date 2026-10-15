# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""The pop-up charts tools show, in the Tools window's look.

A tool the Tools window starts runs in a process of its own, without the
window's QApplication. Its matplotlib chart still takes the window's look:

* its theme colour (tools_theme_color), read from app_settings.json;
* its language (chart_translation), asked of the same catalog;
* the bundled Noto Sans and the window's text colours (plot_font_families,
  chart_style);
* and, in a Qt window, the window's style (show_chart): the Studio palette,
  stylesheet and fonts, the Tools icon, and Lucide icons on matplotlib's
  toolbar.

Nothing here imports matplotlib or Qt until a chart is drawn: headless (JSON
or MCP) runs print their results instead.
"""

import os

# The Tools window keeps its theme colour in app_settings.json under this
# setting (Desktop_App's TOOLS_THEME_SETTING). It launches the tools, so their
# pop-up charts take that colour.
TOOLS_THEME_SETTING = "TOOLS_THEME_COLOR"

# The Lucide icon each of matplotlib's Qt toolbar actions shows, by the
# action's callback name (NavigationToolbar2QT.toolitems).
TOOLBAR_ICONS = {
    "home": "house",
    "back": "arrow-left",
    "forward": "arrow-right",
    "pan": "move",
    "zoom": "zoom-in",
    "configure_subplots": "sliders-horizontal",
    "edit_parameters": "square-pen",
    "save_figure": "save",
}
# The toolbar's icons, in logical pixels: the size of the windows' icon buttons (icon_button).
TOOLBAR_ICON_SIZE = 16


def tools_theme_color():
    """The Tools window's theme colour, or the default one when it can't be read."""
    from desktop.Studio_Theme import DEFAULT_THEME_COLOR, THEME_COLORS
    from utilities.App_Settings import AppSettingsError, read_app_settings

    try:
        color = read_app_settings().get(TOOLS_THEME_SETTING)
    except AppSettingsError:
        color = None
    return color if isinstance(color, str) and color in THEME_COLORS else DEFAULT_THEME_COLOR


def chart_translation(chart):
    """(language, translator) for a chart: the Tools window's language, or (None, None).

    The language is the one the windows show: the saved LANGUAGE setting,
    else the system's, or the pseudo-language for tests
    (Desktop_App.startup_language). The translator answers from that
    language's catalog (Desktop_App.message_translator). Anything that
    can't be read leaves the chart English, with a note naming chart: the run
    never fails over its language. The terminal stays English either way.
    """
    try:
        from desktop.Desktop_App import message_translator, startup_language

        language = startup_language()
        return language, message_translator(language)
    except Exception as error:  # A chart in English beats a failed run.
        print(f"Note: the {chart} stays English ({error}).")
        return None, None


def plot_font_families(project_root, language=None):
    """The bundled Noto Sans, registered with matplotlib, else matplotlib's own font.

    The fonts are project_root's. A language whose script Noto Sans lacks
    adds its language pack's font (Noto Sans SC for Simplified Chinese),
    which matplotlib draws those characters in. Otherwise only one family is
    named: matplotlib looks up every listed family for each text, and warns
    when the fallback lacks the title's semibold face.
    """
    from matplotlib import font_manager

    fonts = os.path.join(project_root, "src", "resources", "fonts", "desktop")
    known = {entry.fname for entry in font_manager.fontManager.ttflist}

    def registered(paths):
        found = False
        for path in paths:
            try:
                if path not in known:
                    font_manager.fontManager.addfont(path)
                found = True
            except (OSError, RuntimeError, ValueError):
                pass
        return found

    core = registered(
        os.path.join(fonts, "noto", "NotoSans", face)
        for face in ("NotoSans-Regular.ttf", "NotoSans-SemiBold.ttf", "NotoSans-Bold.ttf")
    )
    families = ["Noto Sans"] if core else ["DejaVu Sans"]
    font = None
    if language is not None:
        try:
            from utilities.Language_Packs import read_pack

            font = read_pack(language).font
        except Exception:  # No usable pack: the core font, as for English.
            font = None
    if font is not None and registered(os.path.join(fonts, path) for path in font.files):
        families.append(font.family)
    return families


def title_weight(families):
    """The weight of a chart's title in the font families plot_font_families gave.

    A language font has regular and bold faces only (Noto Sans SC): asked for
    semibold, matplotlib logs a fallback and draws its characters bold anyway.
    So its title is bold, which Noto Sans has too; English keeps semibold.
    """
    return "semibold" if len(families) == 1 else "bold"


def chart_style(tokens, families):
    """matplotlib's rc settings for a chart in the theme tokens' greys and the font families."""
    return {
        "font.family": families,
        "font.size": 10,
        "axes.edgecolor": tokens["border"],
        "axes.labelcolor": tokens["text_muted"],
        "xtick.color": tokens["border"],
        "ytick.color": tokens["border"],
        "xtick.labelcolor": tokens["text_muted"],
        "ytick.labelcolor": tokens["text_muted"],
    }


def show_chart(figure, color, project_root):
    """Show figure as a pop-up window in the Tools window's look until it is closed, then close it.

    color is the theme colour (a THEME_COLORS name); style_chart_window
    dresses the window.
    """
    import matplotlib.pyplot as plt

    try:
        style_chart_window(figure, color, project_root)
    except Exception as error:  # A plain window beats a failed run.
        print(f"Note: the chart window keeps matplotlib's look ({error}).")
    plt.show()  # Display as a pop-up window
    plt.close(figure)


def style_chart_window(figure, color, project_root):
    """Give figure's Qt window the Tools window's look; True if it had one to style.

    The window takes what the Tools window starts with: the bundled fonts,
    then the Studio palette and stylesheet in color, and the Tools icon from
    project_root. matplotlib's toolbar shows Lucide icons (TOOLBAR_ICONS).
    A backend with no Qt window, such as Agg's, is left alone.
    """
    try:
        from PySide6.QtCore import QSize
        from PySide6.QtGui import QIcon
        from PySide6.QtWidgets import QApplication, QMainWindow
    except ImportError:
        return False
    manager = figure.canvas.manager
    window = getattr(manager, "window", None)
    if not isinstance(window, QMainWindow):
        return False

    from desktop.Desktop_App import configure_qt_application_fonts
    from desktop.Studio_Theme import apply_studio_theme, studio_icon

    application = QApplication.instance()
    configure_qt_application_fonts(application)
    apply_studio_theme(application, color)
    logos = os.path.join(project_root, "src", "bin", "logos")
    for logo in ("tool_logo.ico", "tool_logo.png"):
        if os.path.exists(os.path.join(logos, logo)):
            window.setWindowIcon(QIcon(os.path.join(logos, logo)))
            break

    toolbar = getattr(manager, "toolbar", None)
    if toolbar is not None:
        # Fixed in place, it draws no grip, as no Tools window has one.
        toolbar.setMovable(False)
        toolbar.setIconSize(QSize(TOOLBAR_ICON_SIZE, TOOLBAR_ICON_SIZE))
        for callback, action in getattr(toolbar, "_actions", {}).items():
            if callback in TOOLBAR_ICONS:
                action.setIcon(studio_icon(TOOLBAR_ICONS[callback]))
        # matplotlib sized the window for its own toolbar: keep the figure's size.
        size = figure.canvas.sizeHint()
        window.resize(size.width(), size.height() + toolbar.sizeHint().height())
    return True
