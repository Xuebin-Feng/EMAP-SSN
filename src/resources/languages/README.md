# Translations

This folder holds the catalogs the program's windows take their text from.

| File | What it is |
|---|---|
| `emapssn.ts` | Every text a window can show, in English, without translations. The update command writes it. |
| `emapssn_<language>.ts` | One language's translations, such as `emapssn_de.ts`. Translators edit this file. |
| `emapssn_<language>.qm` | The compiled catalog the program loads. The update command writes it. |
| `Update_Translations.py` | The update command. |

The Config and Tools windows' texts are marked, and so are the Viewer's window (its title, sidebar and dialogs) and its canvas text outside the commands: the HUD, and the console messages of the Viewer, the command engine, the metadata and the agent. Of the commands' own console messages, those of agent, alignment, color, esmfold, export, group, hide, meta, offset, redo, reference, reset, run, save, select, spectrum, undo and zoom are marked; the other commands' and VR Config's texts follow. The tool help pages (`src/tools/tool_descriptions/`) stay English. There is no language catalog yet, so every window shows English.

## Choose the language

The Language dropdown (🌐) sits at the bottom right of the Config, Tools and VR Config windows. It lists:

- **System default**, which follows the operating system's display language when this folder has a catalog for it, and is English otherwise.
- **English**.
- Every language with a compiled catalog here, each named in itself, such as "Deutsch" or "简体中文".

The choice is saved in `app_settings.json` in the project folder, and every window uses it.

- **Config and Tools** redraw at once in the chosen language. They keep their size, position, splitters, tabs, scroll positions and every value entered, saved or not.
- **The Viewer**, with its web pages, and any other window, such as a dialog, uses the language the next time it opens.
- **Number formats** keep following the system's regional settings in every language.

The bundled fonts cover Latin, Greek and Cyrillic. Simplified Chinese brings a
font of its own, loaded only while it shows (see
`src/resources/fonts/desktop/README.md`). In other scripts, Qt windows fall back
to the system's fonts. The Viewer's canvas can't, since it draws each text in
a single face, so a new language in another script needs a bundled font too
(`LANGUAGE_FONTS` in `src/desktop/Desktop_App.py`).

## Update the catalogs

After changing a text that is marked for translation, run this from the project folder:

```
python src/resources/languages/Update_Translations.py
```

The command does four things:

- It collects every marked text.
- It merges the texts into each language's catalog and keeps the translations already made.
- It compiles the catalogs.
- It checks that every translation keeps its text's placeholders.

The test suite fails while `emapssn.ts` does not match the code.

## Start and translate a language

```
python src/resources/languages/Update_Translations.py --add de
pyside6-linguist src/resources/languages/emapssn_de.ts
```

1. Translate in Qt Linguist, which shows the file and line each text comes from.
2. Mark each finished translation as done, and save.
3. Run the update again to compile.

A translation must keep these parts exactly as the English text has them:

- placeholders such as `{count}`, `%n` and `%1`
- markup such as `<b>`
- file patterns such as `*.fasta`

The update reports any translation that changes a placeholder.

## Mark text in the code

Each window files its texts under one context, which translators see as a group: `Config` (shared with VR Config), `Tools` and `Viewer`. `translate` comes from `desktop.Desktop_App`.

| Text | How to mark it |
|---|---|
| A window's text | `translate("Config", "Save")` |
| A counted text | `translate("Config", "%n file(s)", None, count)` |
| Text in a table made before any window, such as a module's labels | `QT_TRANSLATE_NOOP("Config", "Save")` where it is written, and `translate("Config", label)` where it shows |
| A message on the Viewer's console line | `Message("Saved {name}.", name=name)` |
| A counted message on the console line | `Message("Removed %n group(s) from {name}.", n=count, name=name)` |
| A message put together from sentences | `JoinedMessage([Message("Matched %n node(s).", n=nodes), Message("Ignored %n row(s).", n=rows)])` |
| Text from code without Qt that a window shows, such as an error | `ValueError(Message("Enter a profile name."))`, shown with `display_text(error)` |

A `Message` keeps its English text for the terminal, the logs and MCP clients. Only a window shows the translation. `QT_TRANSLATE_NOOP`, `Message` and `JoinedMessage` come from `utilities.Localization`, which code without Qt can import.

A value that is text to translate too, such as a page's name in a message, is a `Message` itself: `Message("{page} opened at {url}", page=Message("Agent UI"), url=url)`. The pseudo-language check can't see an untranslated value, which shows inside the translated text's brackets, so a test checks such a value on its own.

The console line shows only the first line of a message given to `print_help`, and the terminal, which stays English, shows the rest. So a message whose later lines are details for the terminal, such as a list of what is available, marks its first line and adds the details after it: `JoinedMessage([Message("Group '{group}' does not exist.", group=name), details], separator="\n")`. Its English is the same as before. A command's help works the same way: the terminal prints it all, and the console line shows its first line, `Message("Usage: {syntax}", syntax="zoom <width>")`. Command syntax stays English, like the commands typed. Syntax with braces of its own is a value too, since a template's braces are its placeholders: `Message("Error: Spectrum accepts exactly one {syntax}.", syntax="{PROPERTY_NAME}")`.

Never read the console line back (`viewer.console_text.text`) to report or print it: it shows the translation. Keep the message in a variable and give it to both `show_status` and `command_failed`, so the terminal and MCP clients get its English.

Pass a whole sentence as one plain string, and fill in the values after translating it. Don't write `translate("Config", f"Saved {count}")`, and don't add text to a translated one, as in `translate("Config", "Saved") + ":"`. Pass the arguments in order, without names. The update refuses texts that no catalog can list, and calls whose text or count Qt's lupdate would miss, such as a text whose lines a backslash joins. A text that two places show alike in English but a language may not, such as a column called "Count" and the verb, takes a comment telling them apart: `translate("Config", "Count", "statistics column")`.

Keep markup that is not language out of the text where you can: compose Markdown such as `## {heading}` in the code and translate the heading. A dropdown translates its labels, never the values it stores. A choice that shows a name, such as a file, a device or `BLOSUM62`, is marked with `mark_name_item` instead. Code never reads a translated text to decide what to do: a file dialog's chosen filter tells the format by its pattern, such as `*.csv`, which every language keeps.

### Counted texts

A counted text holds `%n`, which shows the count, and marks its English plural endings in brackets right after the word, as in `%n file(s)` or `%n match(es)`. English shows "1 file" and "2 files", in the windows, on the console line and in the terminal. A translation gives one form per plural form of its language, in Qt Linguist. A counted text that a language hasn't translated yet shows in English. Count with `translate`, not `self.tr`, which would show "1 file(s)".

Rephrase a sentence whose other words change with the count, such as "%n file(s) is ready", so that only the marked endings do: "Ready: %n file(s)".

A text has one count. A message with two counts its first with `%n` and fills in the other as a counted `Message` of its own: `Message("Removed %n group(s) from {instances}.", n=len(groups), instances=Message("%n total node instance(s)", n=removed))`.

## Find text that is not marked

Start a window in the test-only pseudo-language:

```
SSN_PSEUDO_TRANSLATION=1 python src/EMAPSSN_Config.py
```

In PowerShell, run `$env:SSN_PSEUDO_TRANSLATION = "1"` first.

Every text from the catalog shows accented, longer and bracketed, like `[Šååṽéé]`.

- Text still in plain English was never marked.
- Text without its closing bracket is cut off.

`tests/test_untranslated_text.py` runs this check for each window. The text on the Viewer's canvas isn't Qt widgets, so the test reads each file's code instead. A file whose every text on the canvas is marked is listed in `CANVAS_MARKED`, and from then on a new unmarked one fails.

## What stays English

These stay English in every language:

- terminal output and logs
- MCP replies
- settings files
- the commands typed in the Viewer's console
