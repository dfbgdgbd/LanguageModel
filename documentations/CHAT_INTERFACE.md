# Chat interface

## Public contract

`Start_SmallLM.bat` is the sole public entry point. It opens
`Main_Run_Program.py`, whose home screen contains exactly two primary choices:

- **Start Chat**
- **Settings**

Settings are in memory only. Every process starts from `RuntimeSettings`
defaults, and no code writes them to a configuration file.

## Settings reference

`SETTING_DEFINITIONS` controls labels, descriptions, validation, ordering, and
allowed ranges. `RuntimeSettings` stores values and translates them into
`AssistantSettings` plus `GenerationSettings`.

| Setting | Default | Allowed | Effect |
| --- | ---: | --- | --- |
| Response backend | `hybrid` | `hybrid`, `transformer`, `retrieval` | selects routing behavior |
| Temperature | `0.75` | `0.0`–`2.0` | `0` uses greedy decoding; higher values increase sampling variety |
| Top-k sampling | `40` | `1`–`200` | number of candidate tokens sampled after temperature scaling |
| Maximum response tokens | `128` | `8`–`256` | transformer generation loop limit |
| Repetition penalty | `1.18` | `1.0`–`2.0` | adjusts logits for tokens generated in the recent window |
| No-repeat phrase size | `4` | `0`–`8` | blocks repeated n-grams; values `0` and `1` effectively disable the check |
| Retrieval threshold | `0.50` | `0.0`–`1.0` | direct-answer threshold in hybrid mode |
| Retrieval candidates | `3` | `1`–`10` | number of ranked documents attached to a response |
| Conversation memory | `8` | `2`–`32` messages | recent role/content messages retained for transformer prompts |
| System prompt | built-in assistant prompt | 1–500 characters | instructions placed after `<system>` |
| Random seed | `42` | `0`–`2,147,483,647` | seeds the PyTorch sampling generator |
| Typing delay | `0.008` | `0.0`–`0.05` seconds/character | UI streaming speed only |
| Minimum thinking time | `0.65` | `0.0`–`5.0` seconds | minimum duration of the thinking indicator |

The settings screen uses a compact three-column layout so 13 settings remain
readable in an 80-column console.

## Chat commands

| Command | Behavior |
| --- | --- |
| `/settings` | opens settings, rebuilds the assistant, and preserves newest history within the new limit |
| `/info` | reads committed report/corpus metadata and displays model statistics without loading transformer weights |
| `/clear` | clears conversation history and redraws the chat screen |
| `/help` | renders the command table |
| `/back` | returns to the home screen and ends the current assistant instance |
| `/quit` | exits the application |

`/model` is accepted as an undocumented alias for `/info`. Plain `back`, `quit`,
and `exit` are also recognized for convenience.

## Response animation

`_thinking_response()` runs `assistant.respond()` on a one-worker thread pool so
Rich can animate status text on the display thread. It waits until both the
model work has finished and the configured minimum thinking time has elapsed.

`_stream_response()` reveals two characters at a time inside a Rich `Live`
panel. The per-chunk sleep is `typing_delay * chunk_length`. A delay of zero
prints the final panel immediately.

The backend label and top retrieval score are displayed in the response panel
subtitle. A tool response has no retrieval document, so its displayed match
score is `0.000`.

## Adding a setting

Every new user-facing runtime field must be wired through all of these places:

1. add a `SettingDefinition` with validation metadata;
2. add the default field to `RuntimeSettings`;
3. map it in `RuntimeSettings.assistant_settings()` or consume it in the UI;
4. decide whether it belongs in the compact live-settings panel;
5. add boundary, reset, and translation assertions in `test_main.py`;
6. update the table above and the public README setting summary.

The test `test_chat_ui_exposes_every_runtime_setting` intentionally fails when
a `RuntimeSettings` field lacks a definition.

## Adding a chat command

1. choose a slash-prefixed command that cannot be confused with a normal query;
2. handle it before `_thinking_response()` in `chat()`;
3. add it to `_show_chat_help()` and, if important, `_chat_header()`;
4. preserve or intentionally reset assistant history;
5. test it through a non-interactive input smoke sequence;
6. update this reference and the public README.

## Terminal compatibility

Keep application-authored decorations ASCII-safe. Rich adapts boxes to legacy
Windows consoles, but unsupported characters in labels can still trigger
encoding failures or replacement glyphs under CP1252. Test at 80 columns and in
both Windows Terminal and a classic redirected console before publishing major
layout changes.
