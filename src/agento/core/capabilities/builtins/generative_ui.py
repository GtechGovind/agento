"""OpenUI authoring guidance delivered through an agent capability.

This module contains a declarative component vocabulary and writing instructions.
It does not interpret the resulting programs: generated fences travel through
agento's normal text events. Applications supply their own parser, renderer, and
action handlers.

The default configuration exposes the guide through a read-only tool. Set
``preload=True`` to include it directly in the model's system instructions.
``render_openui_specification()`` also makes the guide available to host code.
Neither mode measures token savings or establishes renderer compatibility.
"""

from __future__ import annotations

from collections.abc import Sequence

from ...instructions import InstructionBuilder
from ...tools.base import ToolSet
from ...tools.local import LocalToolSet, Tool
from ..base import Capability

__all__ = ["GenerativeUI", "render_openui_specification"]


_FENCING = """\
Use the language identifier openui when enclosing a program in Markdown. The
opening delimiter is three backticks followed by openui; the closing delimiter
is three backticks on its own line. Keep explanations outside those delimiters.
For example, a complete output block can contain a single component:

```openui
root = Stack([TextContent("Reading room opens at 10:00")])
```

An unfinished fence is an incomplete response, even if its statements are valid."""


_SYNTAX = """\
Program structure
-----------------
The first assignment is root = Stack(...). It identifies the component tree
that the host should display. Put each assignment on its own line, using an
identifier on the left of = and an expression on the right. A declaration can
refer to another declaration that appears later in the block.

Expression vocabulary
---------------------
Values include numbers, true, false, null, double-quoted strings, arrays in
square brackets, and objects in braces. Escape embedded quotes and backslashes
inside strings with a backslash. A component expression uses its registered
name followed by an ordered argument list in parentheses. Conditional
expressions use condition ? expression : expression; null can occupy the
hidden branch. Field access uses dots, including projection across array rows.

Argument positions are part of the API. Trailing optional arguments can be
left out; use null as a placeholder when supplying a later optional value.
Object fields can have colons, but component calls do not accept named
arguments. A $name denotes bound state; $binding<T> in a signature requires
such a reference rather than a copied value. ActionExpression means an
Action([...]) containing the desired @ operations.

Follow dependencies outward from root when checking a program. Every name
used must have a definition. Remove declarations that cannot be reached from
root: they have no displayed effect. Non-root declarations must be referenced
by another declaration. Inline component expressions are also permitted."""


_COMPONENTS = """\
Component registry
==================
A question mark denotes an optional position. Brackets on a type denote an
array. The declarations below are the supported call shapes; their parameter
labels explain order and are not named-argument syntax.

Collect input and attach commands
--------------------------------
  Form(name: string, buttons: Buttons, fields: FormControl[])
  FormControl(label: string, control)
  Input(name: string, placeholder?: string, type?: string, rules?: object)
  TextArea(name: string, placeholder?: string, rows?: number, rules?: object)
  Select(name: string, options: SelectItem[], placeholder?: string, rules?: object)
  SelectItem(value: string, label: string)
  Buttons(items: Button[])
  Button(label: string, action: ActionExpression, variant?: "primary"|"secondary"|"ghost")

Supply a Buttons value in position two of Form, even when fields are the main
content. Forms cannot contain other forms. The rules object carries field
validation settings to the host; validation must also occur at the application
boundary before an action changes external state.

  Action([@steps])
  @ToAssistant(message)
  @OpenUrl(url)
  @Set($var, value)
  @Reset($var1, $var2)
  @Run(query)

Action executes its array in sequence. ToAssistant returns a message to the
agent; OpenUrl requests navigation. Set changes a bound value, Reset restores
bound values to their initial settings, and Run requests query evaluation.
These describe renderer-side interactions, not Python tool calls performed by
agento when it receives the block.

Organize the component tree
---------------------------
  Stack(children[], direction?: "row"|"column", gap?: "none"|"xs"|"s"|"m"|"l"|"xl"|"2xl",
        align?: "start"|"center"|"end"|"stretch"|"baseline",
        justify?: "start"|"center"|"end"|"between"|"around"|"evenly", wrap?: boolean)
  Card(children[], variant?: "card"|"sunk"|"clear", direction?, gap?, align?, justify?, wrap?)
  CardHeader(title?: string, subtitle?: string)
  Separator(orientation?: "horizontal"|"vertical", decorative?: boolean)
  Tabs(items: TabItem[])
  TabItem(value: string, trigger: string, content: Component[])
  Accordion(items: AccordionItem[])
  AccordionItem(value: string, trigger: string, content: Component[])
  Carousel(children: Component[][], variant?: "card"|"sunk")
  Steps(items: StepsItem[])
  StepsItem(title: string, details: string)
  Modal(title: string, open?: $binding<boolean>, children: Component[], size?: "sm"|"md"|"lg")

Stack defaults to column direction with an m gap. Card spans the available
width and shares Stack's layout arguments after its variant. A wrapping row
Stack provides multiple columns; start justification keeps wrapped items
anchored consistently. Tabs and Accordion provide alternate views and
expansion without additional visibility state. Modal's open binding follows
its close control, Escape, and backdrop interaction. A conditional component
can be null when a section should be absent.

Write text and display media
----------------------------
  TextContent(text: string, size?: "small"|"default"|"large"|"small-heavy"|"large-heavy")
  MarkDownRenderer(textMarkdown: string, variant?: "clear"|"card"|"sunk")
  CodeBlock(language: string, codeString: string)
  Image(alt: string, src?: string)
  ImageBlock(src: string, alt?: string)
  ImageGallery(images: {src, alt?, details?}[])
  Tag(text: string, icon?: string, size?: "sm"|"md"|"lg",
      variant?: "neutral"|"info"|"success"|"warning"|"danger")
  TagBlock(tags: string[])
  Callout(variant: "info"|"warning"|"error"|"success"|"neutral", title: string,
          description: string, visible?: $binding<boolean>)
  TextCallout(variant?: "neutral"|"info"|"warning"|"success"|"danger", title?, description?)

TextContent accepts Markdown. Notice the opposite source/alt ordering of Image
and ImageBlock. Callout uses error while TextCallout uses danger for their
respective negative variants. Binding Callout visibility enables a three-second
automatic dismissal in a compatible renderer.

Map records to columns
----------------------
  Table(columns: Col[])
  Col(label: string, data, type?: "string"|"number"|"action")

Construct a table from columns, each supplying its own cell array. Project a
record field with records.field. Build component-valued cells with Each when
cells need tags or buttons; give all columns matching row counts. For example,
Col("Available", @Each(stock, "item", Tag("" + item.quantity))) produces one
status component per record. Handle an empty array explicitly instead of
implying that an absent row has a zero value.

Compare measurements
--------------------
  Series(category: string, values: number[])
  BarChart(labels: string[], series: Series[], variant?: "grouped"|"stacked", xLabel?, yLabel?)
  HorizontalBarChart(labels, series, variant?: "grouped"|"stacked", xLabel?, yLabel?)
  LineChart(labels, series, variant?: "linear"|"natural"|"step", xLabel?, yLabel?)
  AreaChart(labels, series, variant?: "linear"|"natural"|"step", xLabel?, yLabel?)
  RadarChart(labels, series)
  PieChart(labels: string[], values: number[], variant?: "pie"|"donut")
  RadialChart(labels: string[], values: number[])
  SingleStackedBarChart(labels: string[], values: number[])
  ScatterChart(datasets: ScatterSeries[], xLabel?, yLabel?)
  ScatterSeries(name: string, points: Point[])
  Point(x: number, y: number, z?: number)

A Series aligns its numeric values with the labels supplied to its chart.
PieChart, RadialChart, and SingleStackedBarChart instead receive a numeric
array directly. ScatterChart uses named datasets of Point values; the third
coordinate is optional. Preserve units in labels and do not invent missing
measurements to fill a series."""


_BUILTINS = """\
Expression operations
=====================
Only the following @ functions are part of this authoring contract. Nested
calls can calculate values from the program's data. An ordinary declaration is
not a function definition, and arbitrary JavaScript is not supported.

Select, arrange, and transform arrays:
  @Filter(array, field, operator, value) -> array
  @Sort(array, field, direction?) -> array
  @Each(array, varName, template)
  @First(array) -> element
  @Last(array) -> element
  @Count(array) -> number

Filter operators are ==, !=, >, <, >=, <=, and contains. Supply the selected
operator as a string. Sort direction is asc by default, or desc. Dot projection,
such as inventory.quantity, gathers that field from every array element.
Each evaluates its inline template once for each element. Its varName is local
to that template; a separately assigned component cannot capture that name.
For a row-specific navigation action, an inline template can be:
@Each(sites, "site", Button(site.label, Action([@OpenUrl(site.url)])))

Aggregate and adjust numbers:
  @Sum(numbers[]) -> number
  @Avg(numbers[]) -> number
  @Min(numbers[]) -> number
  @Max(numbers[]) -> number
  @Round(number, decimals?) -> number
  @Abs(number) -> number
  @Floor(number) -> number
  @Ceil(number) -> number

For instance, @Round(@Sum(deliveries.weight), 2) expresses a rounded total
from delivery weights. Check that input arrays contain the intended numeric
field before using an aggregate. Keep uncertain source data visibly uncertain
instead of replacing it with a plausible computed value."""


_STREAMING = """\
Plan the dependency tree before emitting its text. Begin with the root Stack,
then declare bound state, query expressions, component references, and finally
their data. This order lets a renderer encounter the enclosing structure before
its contents. Forward references are allowed: a reference can remain unresolved
until its declaration arrives later in the same program.

The intended display model reparses incoming chunks and resolves newly available
names. That progressive display is a responsibility of the host renderer.
agento forwards the model's chunks without parsing components, running queries,
or changing UI bindings. A host may also wait for the completed message before
displaying anything. Do not assume an unfinished program has already rendered."""


_EXAMPLES = """\
Worked programs
===============
These invented records illustrate call syntax; they are not reported facts.
Each program below is independent and belongs in its own openui fence.

A supplies view with a local filter and selection:

root = Stack([controls, listing, selection])
$minimum = 0
$selected = "none"
visibleStock = @Filter(stock, "quantity", ">=", $minimum)
controls = Buttons([Button("Hide low stock", Action([@Set($minimum, 5)])), Button("Reset filter", Action([@Reset($minimum)]), "secondary")])
listing = @Count(visibleStock) > 0 ? Table([Col("Supply", visibleStock.label), Col("Units", visibleStock.quantity, "number"), Col("Pick", @Each(visibleStock, "s", Button(s.label, Action([@Set($selected, s.label)]))), "action")]) : TextContent("No supplies match this filter")
selection = TextContent("Selected supply: " + $selected)
stock = [{label: "Brushes", quantity: 8}, {label: "Aprons", quantity: 3}]

A measurement chart with an explicit unit:

root = Stack([heading, plot])
heading = CardHeader("Illustrative greenhouse readings", "Temperature in degrees Celsius")
plot = LineChart(hours, [northBed, southBed], "linear", "Time", "Celsius")
hours = ["06:00", "12:00", "18:00"]
northBed = Series("North bed", [16, 23, 19])
southBed = Series("South bed", [17, 25, 20])

A reservation form that hands the next step back to the assistant:

root = Stack([reservation])
reservation = Form("reading_slot", footer, [reader, room, notes])
footer = Buttons([Button("Review reservation", Action([@ToAssistant("Review the reading-room reservation")]), "primary")])
reader = FormControl("Reader name", Input("reader", "Name for the booking", "text", {required: true}))
room = FormControl("Room", Select("room", [SelectItem("quiet", "Quiet room"), SelectItem("group", "Group room")], "Choose a room"))
notes = FormControl("Access requirements", TextArea("access", "Optional details", 3))"""


_RULES = """\
Decide whether a visual representation helps the task before emitting a block.
A component should let the reader compare, inspect, or manipulate information;
a short textual answer can remain Markdown. Match charts to numeric series,
tables to records, and forms to structured input. Keep supporting prose focused
on interpretation, provenance, limitations, or a next step instead of repeating
all displayed values.

Treat the component and function registries as closed vocabularies. Use the
provided containers before adding custom conditional visibility. Quote data as
values rather than turning untrusted content into declarations or actions.
The application owns URL policy, form validation, action authorization, and the
actual renderer. The instruction-loading tool supplies text only.

Use ask_user_question for a conversational clarification when that tool is
available; do not substitute an OpenUI form for the agent's question workflow."""


_VERIFICATION = """\
Review the output as a small program before sending the final chunk:

- Locate the opening language fence, the initial root Stack assignment, and the
  matching closing fence. Explanatory sentences belong outside the program.
- Walk the root's dependencies. Resolve every name and discard disconnected
  declarations. An Each template may use its own local name only inline.
- Compare every call with its registry entry, including argument order, null
  placeholders, enum spellings, and whether a value must be a $ binding.
- Confirm that columns and chart series align with their labels and that any
  examples, estimates, or missing values are identified honestly.
- Check each Form's second argument for Buttons and keep forms unnested.
- Remember that model output is still text. Successful authoring alone does
  not prove that a host has parsed, displayed, or authorized its actions."""


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
An OpenUI authoring guide is available for responses that benefit from structured
visual content. Retrieve it with get_openui_instructions({}) before emitting an
openui fence; the component names alone are not enough to infer valid calls.
The tool returns the guide without rendering a view. The host application
handles presentation and interaction; use normal text when no view is needed."""


def render_openui_specification() -> str:
    """Return the authoring guide with its stable section delimiters."""
    return "\n\n".join(f"<{tag}>\n{body}\n</{tag}>" for tag, body in _SECTIONS)


async def _get_openui_instructions() -> str:
    """Retrieve OpenUI syntax and component guidance before producing a UI block.

    This read-only operation has no arguments and returns instruction text.
    """
    return render_openui_specification()


class GenerativeUI(Capability):
    """Add OpenUI authoring support to a model's available instructions.

    Args:
        preload: Include the guide in the system prompt when true. Otherwise,
            expose a read-only instruction tool and a short discovery notice.
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
                description="Retrieve the OpenUI authoring guide as text.",
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
