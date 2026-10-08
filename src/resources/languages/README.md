# Translations

This folder holds the catalogs the program's windows take their text from.

| File | What it is |
|---|---|
| `emapssn.ts` | Every text a window can show, in English, without translations. The update command writes it. |
| `emapssn_<language>.ts` | One language's translations, such as `emapssn_de.ts`. Translators edit this file. |
| `emapssn_<language>.qm` | The compiled catalog the program loads. The update command writes it. |
| `Update_Translations.py` | The update command. |

No text is marked for translation yet, so `emapssn.ts` is empty and there is no language catalog. Every window shows English.

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

| Text | How to mark it |
|---|---|
| A widget's text | `self.tr("Save")` |
| Text shared by windows, or outside a widget | `QCoreApplication.translate("Config", "Save")` |
| A counted text | `self.tr("%n file(s)", "", count)` |
| A message on the Viewer's console line | `Message("Saved {count} nodes to {name}.", count=count, name=name)` |

A `Message` keeps its English text for the terminal, the logs and MCP clients. Only the Viewer's console line shows the translation.

Pass a whole sentence as one plain string, and fill in the values after translating it. Don't write `self.tr(f"Saved {count}")`, and don't add text to a translated one, as in `self.tr("Saved") + ":"`. The update refuses texts that no catalog can list.

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
