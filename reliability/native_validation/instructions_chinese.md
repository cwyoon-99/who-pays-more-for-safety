# Native-Speaker Prompt Validation — Annotator Instructions

You are evaluating the quality of **machine-translated prompts** (English → Chinese).
For each row, please rate two dimensions on a 1–5 scale and (optionally) leave a short note.

## Columns to fill

- **`naturalness`** (1–5): Does the translated text read naturally to a Chinese native speaker?
- **`fidelity`** (1–5): Does the translation faithfully preserve the meaning of the English source, including any cultural / contextual nuance?
- **`notes`** (optional): Brief comment if you marked a low score, e.g. "literal translation, sounds awkward".

Please **enter integers 1–5** in `naturalness` and `fidelity`. Leave them blank only if you cannot rate.

## Scale: Naturalness

| Score | Meaning |
|---|---|
| 1 | Very unnatural — sounds clearly machine-translated, hard to understand |
| 2 | Unnatural — awkward grammar or expression, comprehension difficult |
| 3 | Acceptable — meaning conveyed but phrasing is awkward |
| 4 | Mostly natural — minor awkwardness only |
| 5 | Fully natural — reads as if a native speaker wrote it |

## Scale: Semantic / Cultural Fidelity

| Score | Meaning |
|---|---|
| 1 | Meaning seriously distorted or lost |
| 2 | Only part of the core meaning preserved |
| 3 | Core meaning preserved, but cultural / contextual nuance is missing or incorrect connotation |
| 4 | Meaning preserved with only minor cultural drift |
| 5 | Meaning, nuance, and cultural context all preserved accurately |

## Notes

- Time estimate: about 1–2 minutes per row → ~50–100 minutes total for 50 rows.
- Treat each row independently; you do not need to compare across rows.
- The column `chinese_to_korean_reference` provides a Korean rendering of the Chinese translation, prepared by a careful human translator. This is an **auxiliary reference** in case it helps you cross-check meaning; your fidelity score should be based on comparing the **English source** to the **Chinese translation** directly (not to the Korean reference).
- Please **do not modify** any pre-filled column (id, benchmark, category, english_source, translated_prompt, chinese_to_korean_reference).

## Saving

When done, save the file (CSV / Google Sheet) and notify the team. Thank you!
