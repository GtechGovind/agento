"""Generative UI — letting the agent render components instead of markdown.

Markdown cannot express a sortable table wired to a filter, a chart, a form, or a
modal. This capability teaches the model a small declarative language, **openui**,
and it emits a fenced block::

    ```openui
    root = Stack([title, chart])
    title = TextContent("Q4 Revenue", "large-heavy")
    chart = BarChart(labels, [s1], "grouped")
    labels = ["Oct", "Nov", "Dec"]
    s1 = Series("Revenue", [120, 150, 180])
    ```

**agento renders nothing.** No UI code ships here — the block arrives in the
stream as ordinary assistant text, and your application parses and renders it.
That keeps agento an SDK: a React app, a Slack surface and a terminal client can
each render the same output in their own way, or ignore it entirely.

The language is deliberately the one TrueForge's chat UI speaks, so an existing
``@truefoundry/trueforge-ui`` renderer works against agento unchanged. If you are
writing your own renderer, this module *is* the specification — the grammar,
component signatures and built-in functions below are the complete contract.

Two modes:

``preload=True``
    The full specification goes into every system prompt. Roughly 4,000 tokens,
    always present. Right when nearly every response is visual.

``preload=False`` (default)
    The prompt gets three lines saying the capability exists and that
    ``get_openui_instructions`` must be called before writing a block. Costs
    almost nothing until used, at the price of one extra tool call on the turns
    that use it.

Derived from TrueForge's OpenUI prompt (MIT licensed) so that renderers stay
compatible.
"""

from __future__ import annotations

from collections.abc import Sequence

from ...instructions import InstructionBuilder
from ...tools.base import ToolSet
from ...tools.local import LocalToolSet, Tool
from ..base import Capability

__all__ = ["GenerativeUI", "render_openui_specification"]


_FENCING = """\
Every openui program must be inside a fenced block:

```openui
root = Stack([...])
```

The fence must be opened and closed. Nothing else goes inside it."""


_SYNTAX = """\
1. One statement per line: `identifier = Expression`
2. `root` is the entry point. Every program must define `root = Stack(...)`.
3. Expressions are: strings ("..."), numbers, booleans (true/false), null,
   arrays ([...]), objects ({...}), or component calls TypeName(arg1, arg2, ...)
4. Define a name on one line and reference it later — this reads better and
   streams better.
5. EVERY variable except `root` must be referenced by at least one other
   variable. An unreferenced variable is silently dropped and will not render.
6. Arguments are POSITIONAL. Write `Stack([children], "row", "l")`, NOT
   `Stack([children], direction: "row", gap: "l")`. Named-argument syntax is not
   supported and fails silently.
7. Optional arguments may be omitted from the end.
8. Strings use double quotes, with backslash escaping."""


_COMPONENTS = """\
Arguments marked ? are optional. Sub-components may be inline or referenced;
prefer references, which stream better.
Props typed `ActionExpression` take Action([@steps...]).
Props typed `$binding<type>` take a `$variable` reference for two-way binding.

Layout:
  Stack(children[], direction?: "row"|"column", gap?: "none"|"xs"|"s"|"m"|"l"|"xl"|"2xl",
        align?: "start"|"center"|"end"|"stretch"|"baseline",
        justify?: "start"|"center"|"end"|"between"|"around"|"evenly", wrap?: boolean)
      Flex container. Defaults: direction "column", gap "m".
  Card(children[], variant?: "card"|"sunk"|"clear", direction?, gap?, align?, justify?, wrap?)
      Styled container, always full width. Takes all Stack flex arguments.
  CardHeader(title?: string, subtitle?: string)
  Tabs(items: TabItem[])
  TabItem(value: string, trigger: string, content: Component[])
  Accordion(items: AccordionItem[])
  AccordionItem(value: string, trigger: string, content: Component[])
  Steps(items: StepsItem[])
  StepsItem(title: string, details: string)
  Carousel(children: Component[][], variant?: "card"|"sunk")
  Separator(orientation?: "horizontal"|"vertical", decorative?: boolean)
  Modal(title: string, open?: $binding<boolean>, children: Component[], size?: "sm"|"md"|"lg")
      X, Escape and the backdrop close it automatically.

  - For grid-like layouts use Stack with direction "row" and wrap true.
  - Prefer justify "start" (or omit it) with wrap, for even columns.
  - Show/hide a section with a ternary: $editId != "" ? Card([editForm]) : null

Content:
  TextContent(text: string, size?: "small"|"default"|"large"|"small-heavy"|"large-heavy")
      Supports markdown.
  MarkDownRenderer(textMarkdown: string, variant?: "clear"|"card"|"sunk")
  Callout(variant: "info"|"warning"|"error"|"success"|"neutral", title: string,
          description: string, visible?: $binding<boolean>)
      With a `visible` binding it auto-dismisses after 3 seconds.
  TextCallout(variant?: "neutral"|"info"|"warning"|"success"|"danger", title?, description?)
  Image(alt: string, src?: string)
  ImageBlock(src: string, alt?: string)
  ImageGallery(images: {src, alt?, details?}[])
  CodeBlock(language: string, codeString: string)
  Tag(text: string, icon?: string, size?: "sm"|"md"|"lg",
      variant?: "neutral"|"info"|"success"|"warning"|"danger")
  TagBlock(tags: string[])

Tables (COLUMN-oriented — each Col carries its own data array):
  Table(columns: Col[])
  Col(label: string, data, type?: "string"|"number"|"action")

  - Pluck a field from rows with `data.rows.fieldName`.
  - Styled cells: Col("Status", @Each(rows, "r", Tag(r.status, null, "sm",
      r.status == "open" ? "success" : "danger")))
  - Row actions: Col("Actions", @Each(rows, "r", Button("Edit",
      Action([@Set($showEdit, true), @Set($editId, r.id)]))))
  - Empty state: @Count(rows) > 0 ? Table([...]) : TextContent("No data yet")

Charts (2D):
  BarChart(labels: string[], series: Series[], variant?: "grouped"|"stacked", xLabel?, yLabel?)
  LineChart(labels, series, variant?: "linear"|"natural"|"step", xLabel?, yLabel?)
  AreaChart(labels, series, variant?: "linear"|"natural"|"step", xLabel?, yLabel?)
  RadarChart(labels, series)
  HorizontalBarChart(labels, series, variant?: "grouped"|"stacked", xLabel?, yLabel?)
  Series(category: string, values: number[])

Charts (1D — these take plain numbers, not objects):
  PieChart(labels: string[], values: number[], variant?: "pie"|"donut")
  RadialChart(labels: string[], values: number[])
  SingleStackedBarChart(labels: string[], values: number[])

Charts (scatter):
  ScatterChart(datasets: ScatterSeries[], xLabel?, yLabel?)
  ScatterSeries(name: string, points: Point[])
  Point(x: number, y: number, z?: number)

Forms:
  Form(name: string, buttons: Buttons, fields: FormControl[])
      Always pass buttons. Never nest a Form inside a Form.
  FormControl(label: string, control)
  Input(name: string, placeholder?: string, type?: string, rules?: object)
  TextArea(name: string, placeholder?: string, rows?: number, rules?: object)
  Select(name: string, options: SelectItem[], placeholder?: string, rules?: object)
  SelectItem(value: string, label: string)
  Buttons(items: Button[])
  Button(label: string, action: ActionExpression, variant?: "primary"|"secondary"|"ghost")

Actions:
  Action([@steps])         a sequence of steps run in order
  @ToAssistant(message)    send a message back to the agent
  @OpenUrl(url)            open a URL
  @Set($var, value)        set a bound variable
  @Reset($var1, $var2)     restore variables to their defaults
  @Run(query)              re-run a query"""


_BUILTINS = """\
Built-in functions are prefixed with `@`. These are the ONLY functions
available — do not invent others. Use them instead of hardcoding computed values.

  @Count(array) -> number
  @First(array) -> element
  @Last(array) -> element
  @Sum(numbers[]) -> number
  @Avg(numbers[]) -> number
  @Min(numbers[]) -> number
  @Max(numbers[]) -> number
  @Sort(array, field, direction?) -> array          direction "asc" (default) | "desc"
  @Filter(array, field, operator, value) -> array   operator ==, !=, >, <, >=, <=, contains
  @Round(number, decimals?) -> number
  @Abs(number) / @Floor(number) / @Ceil(number) -> number
  @Each(array, varName, template)                   evaluate template per element

They compose: @Count(@Filter(rows, "status", "==", "open")),
@Round(@Avg(rows.score), 1), @Each(rows, "r", Tag(r.status)).

Array pluck: `rows.field` extracts one field from every row — use it for charts
and table columns.

IMPORTANT @Each rule: the loop variable exists ONLY inside the template
expression, which must be written inline.
  CORRECT: Col("Actions", @Each(rows, "r", Button("Edit", Action([@Set($id, r.id)]))))
  WRONG:   btn = Button("Edit", Action([@Set($id, r.id)]))
           Col("Actions", @Each(rows, "r", btn))     # r is undefined in btn"""


_STREAMING = """\
References may be used before they are defined — the parser resolves them after
the whole program is read.

While streaming, the program is re-parsed on every chunk, so unresolved
references simply appear once their definitions arrive. This produces a
progressive top-down reveal: structure first, data filling in after.

Write statements in this order for the best streaming:
  1. root = Stack(...)        the shell appears immediately
  2. $variable declarations   state ready for bindings
  3. queries                  so components render with data
  4. component definitions
  5. leaf data values

Always write `root = Stack(...)` first."""


_EXAMPLES = """\
Example 1 — table (column-oriented):

root = Stack([title, tbl])
title = TextContent("Top Languages", "large-heavy")
tbl = Table([Col("Language", langs), Col("Users (M)", users, "number"), Col("Year", years, "number")])
langs = ["Python", "JavaScript", "Java", "TypeScript", "Go"]
users = [15.7, 14.2, 12.1, 8.5, 5.2]
years = [1991, 1995, 1995, 2012, 2009]

Example 2 — bar chart:

root = Stack([title, chart])
title = TextContent("Q4 Revenue", "large-heavy")
chart = BarChart(labels, [s1, s2], "grouped")
labels = ["Oct", "Nov", "Dec"]
s1 = Series("Product A", [120, 150, 180])
s2 = Series("Product B", [90, 110, 140])

Example 3 — form with validation:

root = Stack([title, form])
title = TextContent("Contact Us", "large-heavy")
form = Form("contact", btns, [nameField, emailField, msgField])
nameField = FormControl("Name", Input("name", "Your name", "text", { required: true, minLength: 2 }))
emailField = FormControl("Email", Input("email", "you@example.com", "email", { required: true, email: true }))
msgField = FormControl("Message", TextArea("message", "Tell us more...", 4, { required: true, minLength: 10 }))
btns = Buttons([Button("Submit", Action([@ToAssistant("Submit")]), "primary"),
                Button("Cancel", Action([@ToAssistant("Cancel")]), "secondary")])

Example 4 — KPI cards from a list:

root = Stack([cards])
cards = Stack([openCard, doneCard], "row")
openCard = Card([TextContent("Open", "small"),
                 TextContent("" + @Count(@Filter(rows, "status", "==", "open")), "large-heavy")])
doneCard = Card([TextContent("Done", "small"),
                 TextContent("" + @Count(@Filter(rows, "status", "==", "done")), "large-heavy")])"""


_RULES = """\
- Choose the component that fits the content: tables for comparison, charts for
  trends, forms for input, cards for grouped facts.
- Do NOT repeat numbers in prose that are already shown in the block. Text
  outside the block should add what the visual cannot: what the pattern means,
  what to do next, what the data does not show.
- If everything is visible in the components, one line of prose is enough.
- Never use openui to ask the user a question — use the ask_user_question tool.
- Use existing components (Tabs, Accordion, Modal) before inventing show/hide
  patterns with ternaries."""


_VERIFICATION = """\
Before finishing, check:
1. `root = Stack(...)` is the FIRST statement.
2. Every referenced name is defined, and every defined name other than `root` is
   reachable from `root`.
3. Arguments are positional — no `name:` syntax anywhere.
4. The ```openui fence is closed.
5. Forms pass their buttons as the second argument, and no Form is nested."""


_SECTIONS: list[tuple[str, str]] = [
    ("openui-fencing", _FENCING),
    ("openui-syntax", _SYNTAX),
    ("openui-components", _COMPONENTS),
    ("openui-builtins", _BUILTINS),
    ("openui-streaming", _STREAMING),
    ("openui-examples", _EXAMPLES),
    ("openui-rules", _RULES),
    ("openui-verification", _VERIFICATION),
]

_DEFERRED_NOTICE = """\
The Agent can render interactive UI that markdown cannot express — tables wired
to filters, charts, forms, modals — by emitting a fenced ```openui block.

Before writing any ```openui block, the Agent MUST call
get_openui_instructions with {} to load the syntax. Do not guess the component
API.

Use it only when the response is genuinely visual. For ordinary answers, write
markdown."""


def render_openui_specification() -> str:
    """The complete openui specification as one string.

    Returned by ``get_openui_instructions`` in deferred mode, and useful on its
    own if you are writing a renderer and want the contract in one place.
    """
    return "\n\n".join(f"<{tag}>\n{body}\n</{tag}>" for tag, body in _SECTIONS)


async def _get_openui_instructions() -> str:
    """Load the full openui authoring instructions.

    Call this before writing any ```openui block. Pass no arguments.
    """
    return render_openui_specification()


class GenerativeUI(Capability):
    """Teaches the model the openui language.

    Args:
        preload: Put the whole specification in the system prompt. Costs roughly
            4,000 tokens on every call; worth it only when most responses are
            visual. The default defers it behind a tool.
    """

    name = "generative_ui"

    def __init__(self, *, preload: bool = False) -> None:
        self._preload = preload
        self._tools = (
            None
            if preload
            else LocalToolSet(
                "openui",
                [Tool(_get_openui_instructions, name="get_openui_instructions", read_only=True)],
                description="Load the instructions for rendering interactive UI.",
                kind="builtin",
            )
        )

    def tool_sets(self) -> Sequence[ToolSet]:
        return [self._tools] if self._tools is not None else []

    def build_instructions(self, builder: InstructionBuilder) -> None:
        if not self._preload:
            builder.add_section("openui", _DEFERRED_NOTICE)
            return
        section = builder.begin_section("openui")
        for tag, body in _SECTIONS:
            section.add_section(tag, body)
