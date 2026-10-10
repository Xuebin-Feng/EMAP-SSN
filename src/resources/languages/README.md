# Translations

This folder holds the catalogs the program's windows take their text from, and one folder per language.

| File | What it is |
|---|---|
| `emapssn.ts` | Every text a window can show, in English, without translations. The update command writes it. |
| `Update_Translations.py` | The update command. |
| `<language>/` | One language, named by its code, such as `de`, `pt_BR` or `zh_CN`. Everything the language brings is in it. |
| `<language>/emapssn_<language>.ts` | The language's translations, such as `de/emapssn_de.ts`. Translators edit this file. |
| `<language>/emapssn_<language>.qm` | The compiled catalog the program loads. The update command writes it. A folder is offered as a language once this file is in it. |
| `<language>/language.json` | Optional: the language's name, its bundled font and its punctuation (see "Add a language"). |
| `<language>/help/<name>.md` | Optional: translated tool help pages (see "Tool help pages"). |

The Config and Tools windows' texts are marked, and so are the Viewer's window (its title, sidebar and dialogs) and its canvas text outside the commands: the HUD, and the console messages of the Viewer, the command engine, the metadata and the agent. The commands' own console messages are all marked, and so are VR Config's texts and the Agent and metadata pages the Viewer opens in the browser. The tool help pages (`src/tools/tool_descriptions/`) are translated as whole pages. The ESMFold page stays English: it is Mol*'s own interface, which has no translations. The first language is Simplified Chinese (`zh_CN`): every text and every help page is drafted, and a native speaker's review is pending (see "Review a language").

## Choose the language

The Language dropdown (after a globe icon) sits at the bottom right of the Config, Tools and VR Config windows. It lists:

- **System default**, which follows the operating system's display language when this folder has a catalog for it, and is English otherwise.
- **English**.
- Every language with a compiled catalog here, each named in itself, such as "Deutsch" or "简体中文".

The choice is saved in `app_settings.json` in the project folder, and every window uses it.

- **Config and Tools** redraw at once in the chosen language. They keep their size, position, splitters, tabs, scroll positions and every value entered, saved or not.
- **The Viewer**, with its web pages, and any other window, such as a dialog, uses the language the next time it opens.
- **Number formats** keep following the system's regional settings in every language.

The bundled fonts cover Latin, Greek and Cyrillic. Simplified Chinese brings a
font of its own, which its `language.json` names and which loads only while it
shows (see `src/resources/fonts/desktop/README.md`): the Qt windows, the
Viewer's canvas and the browser views (the Agent and metadata pages and the
Tools help panel) all draw Chinese in it. In other scripts, Qt windows fall back
to the system's fonts. The Viewer's canvas can't, since it draws each text in
a single face, so a new language in another script needs a bundled font too
(see "Add a language").

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

### VR Config's own catalog

VR Config files its texts under `Config`, as the desktop Config does, so the texts the two share come from `emapssn.ts`. Its own texts, such as the VR client's fields and notes, are in opt_vr's catalog, `opt_vr/src/resources/languages/emapssn_vr.ts`, which lists only the texts `emapssn.ts` lacks. Update it after the main catalog, since a text the two windows come to share leaves it:

```
python opt_vr/src/resources/languages/Update_Translations_VR.py
```

It keeps a catalog, `<language>/emapssn_vr_<language>.ts` in a folder named as the main one's, for each language the main catalogs have, so a language added here gets one there at its next update; translate the two together. VR Config shows a text its catalog doesn't translate yet in English. It installs the catalog beside the main one, with `install_translations(app, language, extra_catalogs=...)`, and opt_vr's test suite fails while `emapssn_vr.ts` does not match its code.

## Add a language

A language is its folder: adding one means adding `<language>/` here, and nothing in the code names a language. Start it with the update command, which makes the folder and its catalog:

```
python src/resources/languages/Update_Translations.py --add de
python opt_vr/src/resources/languages/Update_Translations_VR.py
pyside6-linguist src/resources/languages/de/emapssn_de.ts opt_vr/src/resources/languages/de/emapssn_vr_de.ts
```

1. Translate in Qt Linguist, which shows the file and line each text comes from.
2. Mark each finished translation as done, and save.
3. Run the updates again to compile. The language shows in the 🌐 dropdown once `de/emapssn_de.qm` is there.
4. Translate the tool help pages you want into `de/help/` (see "Tool help pages"); the others show in English.

A folder made elsewhere works the same way: copy it here whole. Its name is the language's code, written exactly so, as are its files' names: `pt_BR`, not `pt_br`, and `emapssn_pt_BR.ts`. The program compares names as the folder lists them, so a wrong case fails on Windows and macOS just as it does on Linux.

### language.json

A language that needs nothing more than Qt's name for it and the bundled Latin, Greek and Cyrillic fonts needs no `language.json`. Simplified Chinese's, `zh_CN/language.json`, shows all three keys:

```json
{
  "name": "简体中文",
  "font": {
    "family": "Noto Sans SC",
    "vispy_face": "NotoSansSC",
    "files": ["noto/NotoSansSC/NotoSansSC-Regular.ttf", "noto/NotoSansSC/NotoSansSC-Bold.ttf"],
    "web_range": "U+3000-303F, U+3040-30FF, U+3100-312F, U+3200-33FF, U+4E00-9FFF, U+F900-FAFF, U+FF00-FFEF"
  },
  "punctuation": {
    "no_space_after": "。！？；：，、）」』】》",
    "separators": {", ": "，", "; ": "；"},
    "separator_script": "U+2E80-9FFF, U+F900-FAFF, U+FF00-FFEF"
  }
}
```

| Key | What it gives |
|---|---|
| `name` | The language's name in itself, as the dropdown lists it. Without it, Qt's name for the code, which can't tell some languages apart (Qt names Simplified and Traditional Chinese alike). |
| `font` | A bundled font, for a script the core Noto faces lack. `family` is the font's own family name, `vispy_face` the name the Viewer's canvas registers it under, `files` its regular and bold faces in `src/resources/fonts/desktop/` (written with `/`), and `web_range` the CSS unicode-range of the characters the browser views draw in it. The font files stay in the fonts folder, listed in its `SHA256SUMS` with their licence in `LICENSE.fonts`. The font registers only while its language shows, so other languages keep the system's fonts for the same characters. |
| `punctuation` | How a message put together from sentences or list items (`JoinedMessage`) is punctuated. `no_space_after` lists the characters a sentence ends in without a space after it, `separators` a list's separators in the language's own form, and `separator_script` the unicode ranges whose text takes those forms, so a list of Latin file names keeps ", ". Without it, the parts join as given, as English does. |

The update command checks every `language.json` and fails while one can't be used; the program warns and uses its defaults for it. A language written right to left isn't supported yet: its windows, canvas and pages would show it left to right.

### Translate a language

A translation must keep these parts exactly as the English text has them:

A translation must keep these parts exactly as the English text has them:

- placeholders such as `{count}`, `%n` and `%1`
- markup such as `<b>`
- file patterns such as `*.fasta`

The update reports any translation that changes a placeholder.

## Review a language

A draft is a translation left unfinished, as Qt Linguist marks it. lrelease
compiles it all the same, so the windows show the drafts until a reviewer
changes them, and the update counts them apart:

```
emapssn_zh_CN.ts: 0 of 1423 texts translated; 1423 more drafted, awaiting review.
```

To review, open the language's two catalogs in Qt Linguist, correct each text
as needed, mark it finished, save, and run both updates to compile:

```
pyside6-linguist src/resources/languages/zh_CN/emapssn_zh_CN.ts opt_vr/src/resources/languages/zh_CN/emapssn_vr_zh_CN.ts
python src/resources/languages/Update_Translations.py
python opt_vr/src/resources/languages/Update_Translations_VR.py
```

A help page's translation is reviewed in its own file, such as
`src/resources/languages/zh_CN/help/Embedding_MSA.md`. Keep its first line as it
is. `tests/test_tool_help_pages.py` checks that each translation keeps its
English page's headings, code, tables, links and math, and
`tests/test_application_fonts.py` that the bundled font has every character a
translation adds.

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

A `Message` keeps its English text for the terminal, the logs and MCP clients. Only a window shows the translation. A `JoinedMessage` shows its translations in their language's punctuation, which the language's `language.json` gives: in Chinese no space follows a sentence end such as "。", and ", " and "; " beside Chinese text show as "，" and "；". Its English is joined as given. `QT_TRANSLATE_NOOP`, `Message` and `JoinedMessage` come from `utilities.Localization`, which code without Qt can import.

A value that is text to translate too, such as a page's name in a message, is a `Message` itself: `Message("{page} opened at {url}", page=Message("Agent UI"), url=url)`. The pseudo-language check can't see an untranslated value, which shows inside the translated text's brackets, so a test checks such a value on its own.

The console line shows only the first line of a message given to `print_help`, and the terminal, which stays English, shows the rest. So a message whose later lines are details for the terminal, such as a list of what is available, marks its first line and adds the details after it: `JoinedMessage([Message("Group '{group}' does not exist.", group=name), details], separator="\n")`. Its English is the same as before. A command's help works the same way: the terminal prints it all, and the console line shows its first line, `Message("Usage: {syntax}", syntax="zoom <width>")`. Command syntax stays English, like the commands typed. Syntax with braces of its own is a value too, since a template's braces are its placeholders: `Message("Error: Spectrum accepts exactly one {syntax}.", syntax="{PROPERTY_NAME}")`.

Never read the console line back (`viewer.console_text.text`) to report or print it: it shows the translation. Keep the message in a variable and give it to both `show_status` and `command_failed`, so the terminal and MCP clients get its English.

A background job's error reaches the console line as the error itself, so a worker raises its errors holding a Message, and MCP clients get `str(error)`. A worker that runs on a snapshot of the viewer keeps the Message it showed instead of reading the snapshot's console line back, as label's `_label_failed` does.

Pass a whole sentence as one plain string, and fill in the values after translating it. Don't write `translate("Config", f"Saved {count}")`, and don't add text to a translated one, as in `translate("Config", "Saved") + ":"`. Pass the arguments in order, without names. The update refuses texts that no catalog can list, and calls whose text or count Qt's lupdate would miss, such as a text whose lines a backslash joins. A text that two places show alike in English but a language may not, such as a column called "Count" and the verb, takes a comment telling them apart: `translate("Config", "Count", "statistics column")`.

Keep markup that is not language out of the text where you can: compose Markdown such as `## {heading}` in the code and translate the heading. A dropdown translates its labels, never the values it stores. A choice that shows a name, such as a file, a device or `BLOSUM62`, is marked with `mark_name_item` instead. Code never reads a translated text to decide what to do: a file dialog's chosen filter tells the format by its pattern, such as `*.csv`, which every language keeps.

### Counted texts

A counted text holds `%n`, which shows the count, and marks its English plural endings in brackets right after the word, as in `%n file(s)` or `%n match(es)`. English shows "1 file" and "2 files", in the windows, on the console line and in the terminal. A translation gives one form per plural form of its language, in Qt Linguist. A counted text that a language hasn't translated yet shows in English. Count with `translate`, not `self.tr`, which would show "1 file(s)".

Rephrase a sentence whose other words change with the count, such as "%n file(s) is ready", so that only the marked endings do: "Ready: %n file(s)".

A text has one count. A message with two counts its first with `%n` and fills in the other as a counted `Message` of its own: `Message("Removed %n group(s) from {instances}.", n=len(groups), instances=Message("%n total node instance(s)", n=removed))`.

## Web pages

The Viewer's browser pages take their text from the same catalogs, each page under a context of its own: `AgentPage` and `MetadataPage`. `PAGE_CONTEXTS` in `src/web_ui/Page_Texts.py` lists each page's files. A page marks its texts three ways:

| Text | How to mark it |
|---|---|
| An element whose content is one text | `<button data-i18n>Clear Chat</button>`. The content may hold markup, such as `<b>` or `<br>`, which a translation keeps. |
| A `title`, `placeholder`, `aria-label` or `alt` attribute | Nothing: each one that holds a letter is a text. |
| A text a script shows | `t("Agent activated: {model}", {model: name})`, with the English as a plain string literal |

As the Viewer's web server serves a page or one of its scripts, it writes each text's translation in place of its English. So a page needs no catalog of its own, and `t()`, from `src/web_ui/page_text.js`, only fills in the values, as a `Message` does. A page shows the language the Viewer started in, and its `<html lang>` names it.

A page's text can't be counted, so it holds no `%n`. A text that a `.js` file passes to `t()` holds no backslash, since lupdate reads one wrongly there. The update refuses both. A message the Viewer sends a page, such as an error, is a `Message` it gives the page with `display_text`, so the page shows it translated while the terminal and MCP clients get the English.

What a page saves stays English, as the settings files do: the names of the model cards the Agent page starts with or adds, which go to `model_card.json`.

## Tool help pages

The Tools window shows a tab's help from `src/tools/tool_descriptions/<name>.md`. A translation is a whole page in its language's folder, under the English page's name, `<language>/help/<name>.md`, such as `zh_CN/help/Embedding_MSA.md`, whose first line names the English page it was made from:

```
<!-- Translation of Embedding_MSA.md, sha256 <the English page's SHA-256> -->
```

Tools shows the translation in its language while that line matches the English page, and the English page otherwise. So after the English page changes, it shows until its translation is brought up to date. The update command lists such translations, with the first line each needs (`utilities/Help_Pages.py` makes it, as `translation_marker`), and any translation still beside the English pages as `<name>.<language>.md`, where Tools doesn't look; those are notes, not failures. A translation keeps the English page's headings, tool names, setting keys, code and file names, and a tool's heading reads as its card title does in the catalog.

## Find text that is not marked

Start a window in the test-only pseudo-language:

```
SSN_PSEUDO_TRANSLATION=1 python src/EMAPSSN_Config.py
```

In PowerShell, run `$env:SSN_PSEUDO_TRANSLATION = "1"` first.

Every text from the catalog shows accented, longer and bracketed, like `[Šååṽéé]`.

- Text still in plain English was never marked.
- Text without its closing bracket is cut off.

`tests/test_untranslated_text.py` runs this check for each window. The text on the Viewer's canvas isn't Qt widgets, so the test reads each file's code instead. A file whose every text on the canvas is marked is listed in `CANVAS_MARKED`, and from then on a new unmarked one fails. The check follows a name through its whole function, so text that only the terminal or MCP clients get, such as print's full report, goes in a variable of its own, not in one the console line shows.

`tests/test_page_translations.py` opens the Agent and metadata pages in Qt WebEngine under the pseudo-language, and reads each page's code for text that an element or a script shows unmarked.

## What stays English

These stay English in every language:

- terminal output and logs
- MCP replies
- settings files
- the commands typed in the Viewer's console
- the notes the Viewer adds to the agent's replies, such as "No explanatory text was returned.", which the model reads back with the conversation
- the ESMFold page, which is Mol*'s own interface

The benchmark's report is the one file the program writes in the chosen language. `src/resources/benchmark/Run_Benchmark.py` reads the language once, when a run starts, and writes `Benchmark_Report_<date>_<time>.txt` in it; the terminal's copy of the report, the `.json` beside it and the summary an MCP job returns stay English. Its texts are `Message`s under the "Message" context: the script renders them with `display_text` after installing the language in an offscreen `QGuiApplication` (a bare `QCoreApplication` crashes when the Chinese font registers), and removes the language once the file is written. Its columns are padded by display width, so a wide script keeps them lined up.
