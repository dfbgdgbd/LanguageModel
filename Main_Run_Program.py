"""Color terminal interface for chatting with the local SmallLM assistant.

The public launcher is Start_SmallLM.bat. Settings live only in memory and reset
whenever the program exits.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, fields
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Callable


PROJECT_ROOT = Path(__file__).resolve().parent
PROJECT_PYTHON = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"


def _use_project_environment_when_double_clicked() -> None:
    """Relaunch through .venv so a Windows double-click finds dependencies."""
    if os.name != "nt" or not PROJECT_PYTHON.is_file():
        return
    if Path(sys.executable).resolve() == PROJECT_PYTHON.resolve():
        return
    result = subprocess.run([str(PROJECT_PYTHON), str(Path(__file__).resolve())])
    raise SystemExit(result.returncode)


_use_project_environment_when_double_clicked()


try:
    from rich.align import Align
    from rich.box import DOUBLE, ROUNDED
    from rich.console import Console, Group
    from rich.live import Live
    from rich.panel import Panel
    from rich.prompt import Prompt
    from rich.table import Table
    from rich.text import Text
except ModuleNotFoundError as error:  # Friendly message when double-clicked too early.
    if error.name != "rich":
        raise
    print("SmallLM's terminal interface needs the 'rich' package.")
    print("Double-click Start_SmallLM.bat to finish setup and launch SmallLM.")
    if sys.stdin.isatty():
        input("Press Enter to close...")
    raise SystemExit(1) from error


APP_TITLE = "SmallLM Terminal"
DEFAULT_SYSTEM_PROMPT_TEXT = (
    "You are SmallLM, a helpful local assistant. Follow the request, be concise, "
    "and say when you are uncertain."
)


@dataclass(frozen=True)
class SettingDefinition:
    """Metadata for a user-editable runtime setting."""

    key: str
    label: str
    description: str
    minimum: float | None = None
    maximum: float | None = None
    integer: bool = False
    choices: tuple[str, ...] = ()
    text: bool = False


SETTING_DEFINITIONS = (
    SettingDefinition(
        "backend",
        "Response backend",
        "hybrid is recommended; transformer always generates",
        choices=("hybrid", "transformer", "retrieval"),
    ),
    SettingDefinition(
        "temperature",
        "Temperature",
        "lower is focused; higher is more varied",
        0.0,
        2.0,
    ),
    SettingDefinition(
        "top_k",
        "Top-k sampling",
        "number of likely next tokens considered",
        1,
        200,
        integer=True,
    ),
    SettingDefinition(
        "max_new_tokens",
        "Maximum response tokens",
        "upper limit for transformer-generated responses",
        8,
        256,
        integer=True,
    ),
    SettingDefinition(
        "repetition_penalty",
        "Repetition penalty",
        "discourages repeating recently generated tokens",
        1.0,
        2.0,
    ),
    SettingDefinition(
        "no_repeat_ngram_size",
        "No-repeat phrase size",
        "blocks repeated phrases of this token length; 0 disables",
        0,
        8,
        integer=True,
    ),
    SettingDefinition(
        "retrieval_threshold",
        "Retrieval threshold",
        "minimum match score used directly by hybrid mode",
        0.0,
        1.0,
    ),
    SettingDefinition(
        "retrieval_limit",
        "Retrieval candidates",
        "number of local matches considered for each prompt",
        1,
        10,
        integer=True,
    ),
    SettingDefinition(
        "max_history_messages",
        "Conversation memory",
        "recent user and assistant messages retained in context",
        2,
        32,
        integer=True,
    ),
    SettingDefinition(
        "system_prompt",
        "System prompt",
        "instructions that guide generated responses",
        text=True,
    ),
    SettingDefinition(
        "seed",
        "Random seed",
        "reproduces transformer sampling when settings match",
        0,
        2_147_483_647,
        integer=True,
    ),
    SettingDefinition(
        "typing_delay",
        "Typing delay",
        "seconds per displayed character; 0 is instant",
        0.0,
        0.05,
    ),
    SettingDefinition(
        "thinking_delay",
        "Minimum thinking time",
        "minimum seconds to display the thinking animation",
        0.0,
        5.0,
    ),
)


@dataclass
class RuntimeSettings:
    """Session-only settings; this class deliberately has no persistence."""

    backend: str = "hybrid"
    temperature: float = 0.75
    top_k: int = 40
    max_new_tokens: int = 128
    repetition_penalty: float = 1.18
    no_repeat_ngram_size: int = 4
    retrieval_threshold: float = 0.50
    retrieval_limit: int = 3
    max_history_messages: int = 8
    system_prompt: str = DEFAULT_SYSTEM_PROMPT_TEXT
    seed: int = 42
    typing_delay: float = 0.008
    thinking_delay: float = 0.65

    def reset(self) -> None:
        """Restore the launch defaults without reading or writing a file."""
        defaults = type(self)()
        for item in fields(self):
            setattr(self, item.name, getattr(defaults, item.name))

    def set_from_text(self, key: str, raw_value: str) -> None:
        """Validate and apply one value entered in the settings screen."""
        definition = next(
            (item for item in SETTING_DEFINITIONS if item.key == key), None
        )
        if definition is None:
            raise ValueError(f"Unknown setting: {key}")

        stripped_value = raw_value.strip()
        if definition.text:
            if not stripped_value:
                raise ValueError("This setting cannot be empty.")
            if len(stripped_value) > 500:
                raise ValueError("Text settings are limited to 500 characters.")
            setattr(self, key, stripped_value)
            return

        value_text = stripped_value.casefold()
        if definition.choices:
            if value_text not in definition.choices:
                choices = ", ".join(definition.choices)
                raise ValueError(f"Choose one of: {choices}.")
            setattr(self, key, value_text)
            return

        try:
            value: int | float
            value = int(value_text) if definition.integer else float(value_text)
        except ValueError as error:
            expected = "whole number" if definition.integer else "number"
            raise ValueError(f"Enter a valid {expected}.") from error

        if definition.minimum is not None and value < definition.minimum:
            raise ValueError(f"Value must be at least {definition.minimum:g}.")
        if definition.maximum is not None and value > definition.maximum:
            raise ValueError(f"Value must be at most {definition.maximum:g}.")
        setattr(self, key, value)

    def assistant_settings(self):
        """Translate UI values into the existing SmallLM inference settings."""
        from smalllm import AssistantSettings
        from smalllm.backend import GenerationSettings

        return AssistantSettings(
            backend=self.backend,
            direct_retrieval_threshold=self.retrieval_threshold,
            retrieval_limit=self.retrieval_limit,
            max_history_messages=self.max_history_messages,
            system_prompt=self.system_prompt,
            seed=self.seed,
            generation=GenerationSettings(
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_k=self.top_k,
                repetition_penalty=self.repetition_penalty,
                no_repeat_ngram_size=self.no_repeat_ngram_size,
            ),
        )

    def display_value(self, key: str) -> str:
        value = getattr(self, key)
        if key == "system_prompt":
            return value if len(value) <= 48 else f"{value[:45]}..."
        if isinstance(value, float):
            return f"{value:.3f}" if key == "typing_delay" else f"{value:.2f}"
        return str(value)


def model_info_rows() -> tuple[tuple[str, str], ...]:
    """Read model statistics for the chat interface without loading its weights."""
    from smalllm.config import DEFAULT_RETRIEVAL_CORPUS_PATH, DEFAULT_TRAINING_REPORT

    report = json.loads(DEFAULT_TRAINING_REPORT.read_text(encoding="utf-8"))
    config = report["config"]
    with DEFAULT_RETRIEVAL_CORPUS_PATH.open(encoding="utf-8") as corpus:
        retrieval_scenarios = sum(1 for line in corpus if line.strip())
    return (
        ("Model", "decoder-only transformer"),
        ("Parameters", f"{report['parameter_count']:,}"),
        ("Vocabulary", f"{config['vocab_size']:,} tokens"),
        ("Architecture", f"{config['n_layers']} layers / {config['n_heads']} heads"),
        ("Context", f"{config['max_seq_len']} tokens"),
        ("Training pairs", f"{report['corpus_records']:,}"),
        ("Retrieval scenarios", f"{retrieval_scenarios:,}"),
        ("Training steps", f"{report['steps']:,}"),
        ("Validation loss", f"{report['best_validation_loss']:.4f}"),
        ("Validation perplexity", f"{report['best_validation_perplexity']:.2f}"),
        ("Pretrained weights", "none"),
    )


class SmallLMTerminal:
    """Home, settings, and streaming chat screens for SmallLM."""

    def __init__(
        self,
        settings: RuntimeSettings | None = None,
        console: Console | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.settings = settings or RuntimeSettings()
        self.console = console or Console(highlight=False)
        self.sleep = sleep

    @staticmethod
    def _set_window_title() -> None:
        if os.name == "nt":
            try:
                import ctypes

                ctypes.windll.kernel32.SetConsoleTitleW(APP_TITLE)
            except (AttributeError, OSError):
                pass

    def _title_panel(self, subtitle: str) -> Panel:
        title = Text("S M A L L L M", style="bold bright_cyan", justify="center")
        tagline = Text(subtitle, style="bright_white", justify="center")
        return Panel(
            Align.center(Group(title, Text(""), tagline)),
            box=DOUBLE,
            border_style="bright_blue",
            padding=(1, 3),
        )

    @staticmethod
    def _menu_button(number: str, label: str, description: str, color: str) -> Panel:
        content = Group(
            Text(f"[ {number} ]  {label}", style=f"bold {color}", justify="center"),
            Text(description, style="dim", justify="center"),
        )
        return Panel(
            Align.center(content),
            box=ROUNDED,
            border_style=color,
            padding=(1, 2),
        )

    def show_home(self) -> str:
        self.console.clear()
        self.console.print(
            self._title_panel(
                "This is a small ~6m parameter model running on limited training data, "
                "improvements to be made"
            )
        )
        self.console.print()
        buttons = Table.grid(expand=True, padding=(0, 1))
        buttons.add_column(ratio=1)
        buttons.add_column(ratio=1)
        buttons.add_row(
            self._menu_button("1", "START CHAT", "Talk with SmallLM", "bright_green"),
            self._menu_button("2", "SETTINGS", "Tune this session", "bright_magenta"),
        )
        self.console.print(buttons)
        self.console.print()
        self.console.print(
            Align.center(
                Text("Choose 1 or 2  |  Q exits", style="dim bright_white")
            )
        )
        return Prompt.ask(
            "\n[bold bright_cyan]Select[/]",
            choices=["1", "2", "q"],
            default="1",
            console=self.console,
        ).casefold()

    def _settings_table(self) -> Table:
        table = Table(
            box=ROUNDED,
            border_style="bright_magenta",
            header_style="bold bright_white on dark_magenta",
            expand=True,
            row_styles=("", "on grey7"),
        )
        table.add_column("#", style="bold bright_magenta", width=3, justify="center")
        table.add_column("Setting and effect", min_width=28, ratio=2)
        table.add_column("Current", style="bold bright_green", min_width=18, ratio=1)
        for index, definition in enumerate(SETTING_DEFINITIONS, 1):
            setting = Text(definition.label, style="bold bright_cyan")
            setting.append(f"\n{definition.description}", style="dim bright_white")
            table.add_row(
                str(index),
                setting,
                self.settings.display_value(definition.key),
            )
        return table

    def show_settings(self) -> None:
        while True:
            self.console.clear()
            self.console.print(self._title_panel("Session settings"))
            self.console.print()
            self.console.print(self._settings_table())
            self.console.print(
                "\n[bold yellow]These changes are temporary.[/] "
                "Every new launch restores the defaults."
            )
            self.console.print(
                "[dim]Enter a setting number to edit  |  R resets  |  B goes back[/]"
            )
            choice = Prompt.ask("\n[bold bright_magenta]Settings[/]", console=self.console)
            normalized = choice.strip().casefold()
            if normalized in {"b", "back", "q", "quit"}:
                return
            if normalized in {"r", "reset"}:
                self.settings.reset()
                self._notice("Defaults restored.", "bright_green")
                continue
            if not normalized.isdigit() or not 1 <= int(normalized) <= len(
                SETTING_DEFINITIONS
            ):
                self._notice("Choose a listed number, R, or B.", "bright_red")
                continue

            definition = SETTING_DEFINITIONS[int(normalized) - 1]
            current = self.settings.display_value(definition.key)
            if definition.choices:
                new_value = Prompt.ask(
                    f"[bold]{definition.label}[/]",
                    choices=list(definition.choices),
                    default=current,
                    console=self.console,
                )
            elif definition.text:
                new_value = Prompt.ask(
                    f"[bold]{definition.label}[/]",
                    default=str(getattr(self.settings, definition.key)),
                    console=self.console,
                )
            else:
                value_range = f"{definition.minimum:g}-{definition.maximum:g}"
                new_value = Prompt.ask(
                    f"[bold]{definition.label}[/] [dim]({value_range})[/]",
                    default=current,
                    console=self.console,
                )
            try:
                self.settings.set_from_text(definition.key, new_value)
            except ValueError as error:
                self._notice(str(error), "bright_red")

    def _notice(self, message: str, color: str) -> None:
        self.console.print(f"\n[bold {color}]{message}[/]")
        Prompt.ask("[dim]Press Enter to continue[/]", default="", console=self.console)

    def _compact_settings_panel(self) -> Panel:
        details = Table.grid(padding=(0, 1), expand=True)
        details.add_column(style="bright_cyan", width=12, no_wrap=True)
        details.add_column(style="bright_white", justify="right", no_wrap=True)
        details.add_row("Backend", self.settings.backend)
        details.add_row(
            "Temp / top-k", f"{self.settings.temperature:.2f} / {self.settings.top_k}"
        )
        details.add_row("Max tokens", str(self.settings.max_new_tokens))
        details.add_row(
            "Penalty / n",
            f"{self.settings.repetition_penalty:.2f} / {self.settings.no_repeat_ngram_size}",
        )
        details.add_row(
            "Retrieval",
            f"{self.settings.retrieval_limit} @ {self.settings.retrieval_threshold:.2f}",
        )
        details.add_row("Memory", f"{self.settings.max_history_messages} messages")
        details.add_row("Seed", str(self.settings.seed))
        return Panel(
            details,
            title="[bold bright_magenta]LIVE SETTINGS[/]",
            box=ROUNDED,
            border_style="bright_magenta",
            padding=(0, 1),
        )

    def _chat_header(self) -> Table:
        chat_title = Panel(
            Align.center(
                Group(
                    Text("SMALLLM CHAT", style="bold bright_cyan", justify="center"),
                    Text(
                        "Commands:",
                        style="bright_white",
                        justify="center",
                    ),
                    Text(
                        "/settings  |  /info  |  /clear",
                        style="dim",
                        justify="center",
                    ),
                    Text(
                        "/help  |  /back  |  /quit",
                        style="dim",
                        justify="center",
                    ),
                )
            ),
            box=ROUNDED,
            border_style="bright_blue",
            padding=(1, 1),
        )
        layout = Table.grid(expand=True, padding=(0, 1))
        layout.add_column(ratio=1)
        layout.add_column(width=32)
        layout.add_row(chat_title, self._compact_settings_panel())
        return layout

    def _thinking_response(self, assistant: Any, query: str):
        """Generate off the display thread while animated dots remain visible."""
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(assistant.respond, query)
            started = time.monotonic()
            frame = 0
            with self.console.status(
                "[bold bright_magenta]SmallLM is thinking...[/]",
                spinner="line",
                spinner_style="bright_cyan",
            ) as status:
                while (
                    not future.done()
                    or time.monotonic() - started < self.settings.thinking_delay
                ):
                    dots = "." * (frame % 3 + 1)
                    status.update(f"[bold bright_magenta]SmallLM is thinking{dots}[/]")
                    frame += 1
                    self.sleep(0.14)
            return future.result()

    def _response_panel(self, text: str, backend: str, score: float) -> Panel:
        footer = f"{backend}  |  match {score:.3f}"
        return Panel(
            Text(text or " ", style="bright_white"),
            title="[bold bright_magenta] SmallLM [/bold bright_magenta]",
            subtitle=f"[dim]{footer}[/dim]",
            subtitle_align="right",
            box=ROUNDED,
            border_style="bright_magenta",
            padding=(1, 2),
        )

    def _stream_response(self, response: Any) -> None:
        score = response.retrieved[0].score if response.retrieved else 0.0
        text = response.text.strip() or "I do not have a response for that yet."
        if self.settings.typing_delay <= 0:
            self.console.print(self._response_panel(text, response.backend, score))
            return

        visible = ""
        chunk_size = 2
        with Live(
            self._response_panel("", response.backend, score),
            console=self.console,
            refresh_per_second=30,
            transient=False,
            vertical_overflow="visible",
        ) as live:
            for start in range(0, len(text), chunk_size):
                chunk = text[start : start + chunk_size]
                visible += chunk
                live.update(
                    self._response_panel(visible, response.backend, score), refresh=True
                )
                self.sleep(self.settings.typing_delay * len(chunk))

    def _show_chat_help(self) -> None:
        help_table = Table.grid(padding=(0, 2))
        help_table.add_column(style="bold bright_cyan")
        help_table.add_column(style="bright_white")
        help_table.add_row("/settings", "Edit every runtime setting and apply it now")
        help_table.add_row("/info", "Show architecture and training statistics")
        help_table.add_row("/clear", "Clear conversation memory and redraw the screen")
        help_table.add_row("/back", "Return to the main page")
        help_table.add_row("/quit", "Close SmallLM Terminal")
        help_table.add_row("/help", "Show this command list")
        self.console.print(
            Panel(help_table, title="Chat commands", border_style="bright_blue")
        )

    def _show_model_info(self) -> None:
        info = Table.grid(padding=(0, 2))
        info.add_column(style="bold bright_cyan", no_wrap=True)
        info.add_column(style="bright_white")
        for label, value in model_info_rows():
            info.add_row(label, value)
        self.console.print(
            Panel(
                info,
                title="[bold bright_magenta]SmallLM model information[/]",
                border_style="bright_blue",
                box=ROUNDED,
                padding=(1, 2),
            )
        )

    def chat(self) -> bool:
        """Run a chat session. Return True when the entire program should exit."""
        from smalllm import AdvancedAssistant

        self.console.clear()
        self.console.print(self._chat_header())
        with self.console.status(
            "[bold bright_cyan]Loading the local assistant...[/]", spinner="line"
        ):
            assistant = AdvancedAssistant(self.settings.assistant_settings())
        self.console.print(
            "\n[dim]Type a message below. Use /settings to tune the current session.[/]\n"
        )

        while True:
            try:
                query = Prompt.ask(
                    "[bold bright_cyan]You[/] [bright_blue]>[/]",
                    console=self.console,
                ).strip()
            except (EOFError, KeyboardInterrupt):
                self.console.print()
                return False
            if not query:
                continue
            command = query.casefold()
            if command in {"/back", "back"}:
                return False
            if command in {"/quit", "/exit", "quit", "exit"}:
                return True
            if command == "/clear":
                assistant.clear_history()
                self.console.clear()
                self.console.print(self._chat_header())
                self.console.print("\n[bold bright_green]Conversation cleared.[/]\n")
                continue
            if command == "/help":
                self._show_chat_help()
                continue
            if command in {"/info", "/model"}:
                self._show_model_info()
                continue
            if command == "/settings":
                history = list(assistant.history)
                self.show_settings()
                self.console.clear()
                self.console.print(self._chat_header())
                with self.console.status(
                    "[bold bright_cyan]Applying session settings...[/]",
                    spinner="line",
                ):
                    assistant = AdvancedAssistant(self.settings.assistant_settings())
                    assistant.history = history[-self.settings.max_history_messages :]
                self.console.print(
                    "\n[bold bright_green]Settings applied to this chat.[/]\n"
                )
                continue

            self.console.print()
            try:
                response = self._thinking_response(assistant, query)
            except (OSError, ValueError, RuntimeError) as error:
                self.console.print(
                    Panel(
                        Text(str(error), style="bright_red"),
                        title="Could not generate a response",
                        border_style="bright_red",
                    )
                )
                continue
            self._stream_response(response)
            self.console.print()

    def run(self) -> int:
        self._set_window_title()
        while True:
            try:
                choice = self.show_home()
            except (EOFError, KeyboardInterrupt):
                self.console.print()
                return 0
            if choice == "q":
                return 0
            if choice == "2":
                self.show_settings()
            elif self.chat():
                return 0


def main() -> int:
    terminal = SmallLMTerminal()
    try:
        return terminal.run()
    except KeyboardInterrupt:
        terminal.console.print("\n[dim]SmallLM closed.[/]")
        return 0
    except (OSError, ValueError, RuntimeError) as error:
        terminal.console.print(
            Panel(
                Text(str(error), style="bright_red"),
                title="SmallLM could not start",
                border_style="bright_red",
            )
        )
        if sys.stdin.isatty():
            input("Press Enter to close...")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
