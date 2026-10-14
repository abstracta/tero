/**
 * Parses the `{{ Variable }}` placeholders a prompt author writes, and substitutes the values
 * a user supplies for them. Names may contain spaces and arguments may be unquoted, which no
 * off-the-shelf template engine accepts — hence the small scanner at the bottom of this file.
 *
 *   template    := ( placeholder | text )*
 *   placeholder := backslash* "{{" name modifier* "}}"
 *   name        := chars up to the first unescaped "|", trimmed
 *   modifier    := "|" ws identifier ( ws ":" ws arg_list )?
 *   identifier  := [A-Za-z_][A-Za-z0-9_-]*
 *   arg_list    := arg ( ws "," ws arg )*
 *   arg         := quoted | bare
 *   quoted      := '"' ( escape | [^"] )* '"'
 *   bare        := chars up to the next unescaped "," or "|", trimmed
 *   escape      := "\" any
 *
 * An even number of leading backslashes (zero included) means a live placeholder; an odd
 * number escapes it, and the braces are emitted literally with the backslashes halved.
 * Only `\|`, `\,`, `\}` and `\"` are unescaped — every other backslash is left verbatim, so
 * a value like `C:\Users\me` survives a round trip.
 *
 * Two deliberate departures from the original grammar: `bare` needs no `}}` terminator
 * because the outer regex has already stripped it, and a placeholder cannot span newlines.
 * Only the `options:` modifier is implemented; anything else resolves to a text input.
 */

export type PromptVariableInput =
  | { type: 'text' }
  | { type: 'select'; options: string[] }

/** A `{{ name | modifier... }}` placeholder found in a prompt. */
export interface PromptVariable {
  name: string
  input: PromptVariableInput
}

// The inner text is a run of escape pairs and plain characters, so a `\}}` belongs to the
// content and only an unescaped `}}` closes the placeholder. Neither branch matches a newline,
// so a stray `{{` still never swallows the rest of a multi-line prompt.
const PLACEHOLDER_PATTERN = /(\\*)\{\{((?:\\.|[^\\\n])*?)\}\}/
const IDENTIFIER_PATTERN = /^[a-zA-Z_][\w-]*/

// a fresh instance per scan, so the shared lastIndex can never leak between calls
const placeholderRegex = () => new RegExp(PLACEHOLDER_PATTERN, 'g')

export function parsePromptVariables(template: string): PromptVariable[] {
  const variables = new Map<string, PromptVariable>()
  const regex = placeholderRegex()
  let match: RegExpExecArray | null
  while ((match = regex.exec(template))) {
    const [, backslashes, inner] = match
    if (backslashes.length % 2 === 1) continue
    const variable = parsePlaceholder(inner)
    // a nameless placeholder has nothing to label its input with, so it collects no answer
    // and stays in the text verbatim — `renderPromptTemplate` leaves it alone to match
    if (variable.name === '') continue
    const known = variables.get(variable.name)
    // repeated names collapse into a single input; a declared widget wins over a bare
    // mention, regardless of which occurrence came first
    if (!known || (known.input.type === 'text' && variable.input.type !== 'text')) {
      variables.set(variable.name, variable)
    }
  }
  return Array.from(variables.values())
}

export function renderPromptTemplate(template: string, values: Record<string, string>): string {
  return template.replace(placeholderRegex(), (_, backslashes: string, inner: string) => {
    const literalSlashes = '\\'.repeat(Math.floor(backslashes.length / 2))
    const name = parsePlaceholder(inner).name
    if (backslashes.length % 2 === 1 || name === '') {
      return `${literalSlashes}{{${inner}}}`
    }
    return literalSlashes + ownValue(values, name)
  })
}

/**
 * Reads a value without consulting the prototype chain, so a variable named `toString` or
 * `constructor` collects an answer like any other instead of rendering whatever
 * `Object.prototype` exposes for that key.
 */
function ownValue(values: Record<string, string>, name: string): string {
  return Object.prototype.hasOwnProperty.call(values, name) ? values[name] ?? '' : ''
}

export function defaultVariableValues(variables: PromptVariable[]): Record<string, string> {
  // no prototype, so a name like `__proto__` is stored as an ordinary key instead of being
  // swallowed by the setter it would otherwise hit
  const values: Record<string, string> = Object.create(null)
  for (const variable of variables) {
    values[variable.name] = variable.input.type === 'select' ? variable.input.options[0] : ''
  }
  return values
}

function parsePlaceholder(inner: string): PromptVariable {
  const scanner = new Scanner(inner)
  const name = unescapeReserved(scanner.readRaw((c) => c === '|')).trim()
  return { name, input: resolveInput(readModifiers(scanner)) }
}

/**
 * Picks the widget a variable's modifiers ask for. Modifiers that declare nothing usable
 * (an empty `options:`, or one this version does not implement yet) fall back to free text,
 * so an unrecognised prompt still collects a value instead of rendering a dead control.
 */
function resolveInput(modifiers: Modifiers): PromptVariableInput {
  const options = modifiers.options
  return options?.length ? { type: 'select', options } : { type: 'text' }
}

type Modifiers = { options?: string[] }

function readModifiers(scanner: Scanner): Modifiers {
  const modifiers: Modifiers = {}
  while (scanner.consume('|')) {
    scanner.skipWhitespace()
    const name = scanner.readMatch(IDENTIFIER_PATTERN)
    if (!name) continue
    scanner.skipWhitespace()
    const args = scanner.consume(':') ? readArgList(scanner) : []
    // every other modifier is still scanned — skipping it would desynchronise the scanner —
    // and then discarded, so an unimplemented one leaves the variable as free text
    // a repeated modifier keeps the last declaration, matching how the widget renders
    if (name === 'options') modifiers.options = args
  }
  return modifiers
}

function readArgList(scanner: Scanner): string[] {
  const args: string[] = []
  do {
    scanner.skipWhitespace()
    args.push(readArg(scanner))
    scanner.skipWhitespace()
  } while (scanner.consume(','))
  return args.filter((arg) => arg !== '')
}

function readArg(scanner: Scanner): string {
  if (scanner.consume('"')) {
    const quoted = scanner.readRaw((c) => c === '"')
    scanner.consume('"')
    return unescapeReserved(quoted)
  }
  return unescapeReserved(scanner.readRaw((c) => c === ',' || c === '|')).trim()
}

/**
 * Unescapes only the characters the grammar reserves — the argument and modifier separators,
 * the closing brace, and the quote that delimits a quoted argument. Every other backslash is
 * left alone, so text that legitimately contains one (`C:\Users\me`) survives untouched.
 */
function unescapeReserved(text: string): string {
  return text.replace(/\\([|,}"])/g, '$1')
}

class Scanner {
  private pos = 0

  constructor(private readonly text: string) {}

  private get done(): boolean {
    return this.pos >= this.text.length
  }

  private peek(): string {
    return this.text[this.pos] ?? ''
  }

  skipWhitespace(): void {
    while (!this.done && /\s/.test(this.peek())) this.pos++
  }

  consume(char: string): boolean {
    if (this.peek() !== char) return false
    this.pos++
    return true
  }

  readMatch(pattern: RegExp): string {
    const match = pattern.exec(this.text.slice(this.pos))
    if (!match) return ''
    this.pos += match[0].length
    return match[0]
  }

  /** Reads until `stop` matches an unescaped char. Escape sequences are kept verbatim. */
  readRaw(stop: (char: string) => boolean): string {
    let out = ''
    while (!this.done) {
      const char = this.peek()
      if (char === '\\' && this.pos + 1 < this.text.length) {
        out += char + this.text[this.pos + 1]
        this.pos += 2
        continue
      }
      if (stop(char)) break
      out += char
      this.pos++
    }
    return out
  }
}

/*
 * TODO move these to `promptVariables.spec.ts`. The JS workspace has no test runner yet (the
 * backend has pytest; `src/` has neither vitest nor jest), so the cases below are recorded here
 * instead of being executable. They were each verified against this implementation. When adding
 * a runner — vitest with `environment: 'node'` is enough, this module has no DOM or Vue
 * dependency — port them verbatim and delete this block. Note that
 * `src/frontend/tsconfig.app.json` includes `../common/src/**` while excluding only
 * `src/**\/__tests__/*`, so a spec file here also needs that exclude widened or
 * `types: ['vitest/globals']` added, or `vue-tsc --build` will fail on `describe`/`it`.
 *
 * Until then, re-verify by hand when changing the scanner.
 *
 * parsePromptVariables(template) → one entry per distinct name, in order of first appearance
 *
 *   Hello {{ Project }}                          → Project        text
 *   {{ Variable name }}                          → Variable name  text        (spaces kept)
 *   {{ Project | options: Alpha, Beta }}         → Project        select [Alpha, Beta]
 *   {{ Env | options: "dev, staging", prod }}    → Env            select [dev, staging | prod]
 *   {{ Env | options: a\, b, c }}                → Env            select [a, b | c]
 *   {{ Name \| Surname }}                        → Name | Surname text
 *   {{ File | files: .pdf, .docx }}              → File           text        (not implemented)
 *   {{ V | options: }}                           → V              text        (no values)
 *   {{ V | toggle | default: true }}             → V              text        (not implemented)
 *   {{ V | 123bad }}                             → V              text        (bad identifier)
 *   {{ Path | options: "C:\Users\me", other }}   → Path           select [C:\Users\me | other]
 *   {{ Path | options: C:\Users\me, other }}     → Path           select [C:\Users\me | other]
 *   {{ Q | options: "say \"hi\"", bye }}         → Q              select [say "hi" | bye]
 *   {{ Env | options: a\}b, c }}                 → Env            select [a}b | c]
 *   {{ Env | options: a}b, c }}                  → Env            select [a}b | c]
 *   {{ V \}} more }}                             → V }} more      text
 *   {{ P }} {{  P  }} {{ P | options: A, B }}    → P              select [A, B]   (one entry;
 *                                                  a declared widget beats a bare mention,
 *                                                  whichever came first)
 *
 * Parsed as no variable at all, left in the text verbatim:
 *
 *   \{{ Literal }}          escaped placeholder
 *   {{ }}                   no name, so no input could be labelled
 *   {{ | options: A, B }}   no name, however well-formed the modifiers are
 *   {{ Multi\nline }}       placeholders cannot span newlines
 *   {{ oops                 unterminated
 *   {{ V \}}                closing braces escaped, so never terminated
 *
 * renderPromptTemplate(template, values)
 *
 *   ('Hello {{ Project }}', {Project: 'Tero'})                  → 'Hello Tero'
 *   ('\{{ Literal }}', {})                                      → '{{ Literal }}'
 *   ('{{ }} {{ | options: A }}', {})                            → '{{ }} {{ | options: A }}'
 *   ('\\{{ Real }}', {Real: 'v'})                               → '\v'   (backslashes halved)
 *   ('{{ P }} {{ P }}', {P: 'A'})                               → 'A A'  (one answer reused)
 *   ('{{ Missing }}', {})                                       → ''
 *
 * Inherited properties must never leak into a rendered prompt — `{{ toString }}`,
 * `{{ constructor }}`, `{{ __proto__ }}`, `{{ valueOf }}`, `{{ hasOwnProperty }}` and
 * `{{ isPrototypeOf }}` all render as '' against `{}`, and stay '' even when
 * `Object.prototype` has been polluted with a matching key.
 *
 * defaultVariableValues
 *
 *   {{ A }} {{ B | options: x, y }}   → {A: '', B: 'x'}   (a select preselects its first option)
 *   {{ __proto__ }}                   → an ordinary own key that stores and reads back a value
 */
