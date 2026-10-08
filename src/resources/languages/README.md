# Translations

This folder holds the catalogs the program's windows take their text from.

| File | What it is |
|---|---|
| `emapssn.ts` | Every text a window can show, in English, without translations. The update command writes it. |
| `emapssn_<language>.ts` | One language's translations, such as `emapssn_de.ts`. Translators edit this file. |
| `emapssn_<language>.qm` | The compiled catalog the program loads. The update command writes it. |
| `Update_Translations.py` | The update command. |

The Config window's texts are marked; the other windows' follow. There is no language catalog yet, so every window shows English.

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
| Text from code without Qt that a window shows, such as an error | `ValueError(Message("Enter a profile name."))`, shown with `display_text(error)` |

A `Message` keeps its English text for the terminal, the logs and MCP clients. Only a window shows the translation. `QT_TRANSLATE_NOOP` and `Message` come from `utilities.Localization`, which code without Qt can import.

Pass a whole sentence as one plain string, and fill in the values after translating it. Don't write `translate("Config", f"Saved {count}")`, and don't add text to a translated one, as in `translate("Config", "Saved") + ":"`. Pass the arguments in order, without names. The update refuses texts that no catalog can list, and calls whose text or count Qt's lupdate would miss. A text that two places show alike in English but a language may not, such as a column called "Count" and the verb, takes a comment telling them apart: `translate("Config", "Count", "statistics column")`.

### Counted texts

A counted text holds `%n`, which shows the count, and marks its English plural endings in brackets right after the word, as in `%n file(s)` or `%n match(es)`. English shows "1 file" and "2 files", in the windows, on the console line and in the terminal. A translation gives one form per plural form of its language, in Qt Linguist. A counted text that a language hasn't translated yet shows in English. Count with `translate`, not `self.tr`, which would show "1 file(s)".

Rephrase a sentence whose other words change with the count, such as "%n file(s) is ready", so that only the marked endings do: "Ready: %n file(s)".

## Find text that is not marked

Start a window in the test-only pseudo-language:

```
SSN_PSEUDO_TRANSLATION=1 python src/EMAPSSN_Config.py
```

In PowerShell, run `$env:SSN_PSEUDO_TRANSLATION = "1"` first.

Every text from the catalog shows accented, longer and bracketed, like `[Šååṽéé]`.

- Text still in plain English was never marked.
- Text without its closing bracket is cut off.

`tests/test_untranslated_text.py` runs this check for each window.

## What stays English

These stay English in every language:

- terminal output and logs
- MCP replies
- settings files
- the commands typed in the Viewer's console
