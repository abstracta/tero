import MarkdownIt from 'markdown-it'
import hljs from 'highlight.js'
import { parseHTML } from 'linkedom'
import * as echarts from 'echarts'
// @ts-ignore
import MarkdownItPlantuml from 'markdown-it-plantuml'

export async function renderMarkdown(markdown: string): Promise<string> {
  const md = createMarkdownRenderer()
  let html = md.render(markdown)
  html = renderEchartsToSvg(html)
  html = await inlineRemoteImages(html)
  return wrapHtml(html)
}

function createMarkdownRenderer(): MarkdownIt {
  const md = new MarkdownIt({
    breaks: true,
    highlight(str, lang) {
      if (!lang) return ''
      const langLabel = `<div class="code-header"><span>${lang}</span></div>`
      let code: string
      if (hljs.getLanguage(lang)) {
        try {
          code = hljs.highlight(str, { language: lang }).value
        } catch {
          code = md.utils.escapeHtml(str)
        }
      } else {
        code = md.utils.escapeHtml(str)
      }
      return `<pre>${langLabel}<code>${code}</code></pre>`
    }
  })

  md.use(MarkdownItPlantuml)
  md.use(useEcharts)
  md.use(useTables)
  md.use(useListDecimal)
  md.use(useListBullet)
  md.use(useBlockquote)
  md.use(useCodeInline)

  return md
}

function useFencePlaceholder(md: MarkdownIt, lang: string, className: string, dataAttr: string): void {
  const defaultRender = md.renderer.rules.fence || ((tokens, idx, options, _env, self) => self.renderToken(tokens, idx, options))

  md.renderer.rules.fence = (tokens, idx, options, env, self) => {
    const token = tokens[idx]
    if (token.info.trim() === lang) {
      const code = token.content.trim()
      return `<div class="${className}" ${dataAttr}="${md.utils.escapeHtml(code)}"></div>`
    }
    return defaultRender(tokens, idx, options, env, self)
  }
}

function useEcharts(md: MarkdownIt): void {
  useFencePlaceholder(md, 'echarts', 'echarts-placeholder', 'data-options')
}

function useTables(md: MarkdownIt): void {
  md.renderer.rules.td_open = () => '<td>'
  md.renderer.rules.th_open = () => '<th>'
  md.renderer.rules.table_open = () => '<div class="table-wrapper"><table>'
  md.renderer.rules.table_close = () => '</table></div>'
}

function useListDecimal(md: MarkdownIt): void {
  const defaultRender = md.renderer.rules.ordered_list_open || ((tokens, idx, options, _env, self) => self.renderToken(tokens, idx, options))
  md.renderer.rules.ordered_list_open = (tokens, idx, options, env, self) => {
    tokens[idx].attrJoin('class', 'list-decimal')
    return defaultRender(tokens, idx, options, env, self)
  }
}

function useListBullet(md: MarkdownIt): void {
  const defaultRender = md.renderer.rules.bullet_list_open || ((tokens, idx, options, _env, self) => self.renderToken(tokens, idx, options))
  md.renderer.rules.bullet_list_open = (tokens, idx, options, env, self) => {
    tokens[idx].attrJoin('class', 'list-disc')
    return defaultRender(tokens, idx, options, env, self)
  }
}

function useBlockquote(md: MarkdownIt): void {
  md.renderer.rules.blockquote_open = () => '<blockquote>'
  md.renderer.rules.blockquote_close = () => '</blockquote>'
}

function useCodeInline(md: MarkdownIt): void {
  md.renderer.rules.code_inline = (tokens, idx) => {
    return `<code class="inline-code">${md.utils.escapeHtml(tokens[idx].content)}</code>`
  }
}

function renderEchartsToSvg(html: string): string {
  return html.replace(/<div class="echarts-placeholder" data-options="([^"]*?)"><\/div>/g, (_match, escaped) => {
    const optionsJson = unescapeHtml(escaped)
    try {
      const options = JSON.parse(optionsJson)
      const svg = renderChartToSvg(options)
      return `<div class="chart">${svg}</div>`
    } catch {
      return `<pre><code>${escaped}</code></pre>`
    }
  })
}

function renderChartToSvg(options: Record<string, unknown>): string {
  const { document } = parseHTML('<body><div id="c" style="width:800px;height:400px;"></div></body>')
  const container = document.getElementById('c')!
  const chart = echarts.init(container as unknown as HTMLElement, null, {
    ssr: true,
    renderer: 'svg',
    width: 800,
    height: 400
  })
  chart.setOption(options)
  const svg = chart.renderToSVGString()
  chart.dispose()
  return svg
}

async function fetchImageDataUri(url: string): Promise<string | null> {
  try {
    const resp = await fetch(url)
    if (!resp.ok) return null
    const contentType = resp.headers.get('content-type') || 'image/png'
    const buffer = Buffer.from(await resp.arrayBuffer())
    return `data:${contentType};base64,${buffer.toString('base64')}`
  } catch {
    return null
  }
}

async function inlineRemoteImages(html: string): Promise<string> {
  const regex = /<img([^>]*?)src="(https?:\/\/[^"]+)"([^>]*?)>/g
  const matches = [...html.matchAll(regex)]
  if (!matches.length) return html

  for (const match of matches) {
    const dataUri = await fetchImageDataUri(match[2])
    if (!dataUri) continue
    html = html.replace(match[0], `<img${match[1]}src="${dataUri}"${match[3]}>`)
  }
  return html
}

function unescapeHtml(str: string): string {
  return str
    .replace(/&amp;/g, '&')
    .replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>')
    .replace(/&quot;/g, '"')
    .replace(/&#39;/g, "'")
}

function wrapHtml(body: string): string {
  return `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<style>${CSS}</style>
</head>
<body>
<div class="chat-container">
  <div class="messages">
    <div class="message agent-message">
      <div class="formatted-text agent-body">${body}</div>
    </div>
  </div>
</div>
</body>
</html>`
}

const CSS = `
:root { --bg: #f4f4f4; --border: #d9d9d9; --muted: #737475; --text: #1f1f1f; --primary: #754bde; --font: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; --mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; }
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html { font-family: var(--font); }
body { background: #fff; color: var(--text); line-height: 1.5; }
a { color: var(--primary); text-decoration: underline; font-weight: 600; }
b, strong { font-weight: 500; }

.chat-container { max-width: 837px; margin: 0 auto; padding: 1rem; }

.messages { display: flex; flex-direction: column; gap: 0.25rem; }
.message { padding: 0.5rem; }
.agent-message { display: flex; flex-direction: column; gap: 0.5rem; }
.agent-body { display: flex; flex-direction: column; gap: 0.5rem; overflow-x: auto; }

.formatted-text img {
  display: block;
  max-width: min(100%, 800px);
  width: auto;
  height: auto;
  max-height: 480px;
  object-fit: contain;
  margin: 0.5rem auto;
  border-radius: 0.5rem;
}
.formatted-text { line-height: 1.375; }
.formatted-text hr { display: none; }
.formatted-text h1, .formatted-text h2 { font-size: 1.25rem; margin: 0.625rem 0; }
.formatted-text h3, .formatted-text h4 { margin: 0.125rem 0; font-weight: normal; }
.formatted-text ul, .formatted-text ol { margin-bottom: 0.625rem; }
.formatted-text p { margin: 0.375rem 0; }
.formatted-text li { margin-left: 1rem; margin-top: 0.5rem; margin-bottom: 0.5rem; }
.formatted-text ul li ul li { list-style-type: circle; }
.formatted-text p br { display: block; margin: 0.625rem 0; }
ol.list-decimal { list-style-type: decimal; padding-left: 1.5rem; }
ul.list-disc { list-style-type: disc; padding-left: 0.5rem; }

code.inline-code { background: var(--bg); padding: 0.125rem; font-size: 0.875rem; font-family: var(--mono); }

pre { margin: 0.75rem 0; }
.code-header { display: flex; align-items: center; padding: 0.5rem; background: var(--bg); border: 1px solid var(--border); border-bottom: none; border-radius: 0.5rem 0.5rem 0 0; }
.code-header span { font-size: 0.75rem; text-transform: lowercase; color: var(--text); font-family: var(--font); }
pre > code { display: block; padding: 0.5rem; background: var(--bg); border: 1px solid var(--border); border-radius: 0 0 0.5rem 0.5rem; overflow-x: auto; font-family: var(--mono); font-size: 0.875rem; line-height: 1.5; }

.table-wrapper { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; }
td, th { border: 1px solid var(--border); padding: 0.5rem 1rem; word-break: break-word; }
th { background: var(--bg); font-weight: 500; }

blockquote { display: flex; gap: 0.75rem; margin: 0.5rem 0; padding: 1.5rem; border-left: 8px solid var(--bg); }

.chart { margin: 1rem 0; display: flex; justify-content: center; }
.chart svg { max-width: 100%; height: auto; }

.hljs{color:#2f3337;background:var(--bg)}.hljs-subst{color:#2f3337}.hljs-comment{color:#656e77}.hljs-attr,.hljs-doctag,.hljs-keyword,.hljs-meta .hljs-keyword,.hljs-section,.hljs-selector-tag{color:#015692}.hljs-attribute{color:#803378}.hljs-name,.hljs-number,.hljs-quote,.hljs-selector-id,.hljs-template-tag,.hljs-type{color:#b75501}.hljs-selector-class{color:#015692}.hljs-link,.hljs-regexp,.hljs-selector-attr,.hljs-string,.hljs-symbol,.hljs-template-variable,.hljs-variable{color:#54790d}.hljs-meta,.hljs-selector-pseudo{color:#015692}.hljs-built_in,.hljs-literal,.hljs-title{color:#b75501}.hljs-bullet,.hljs-code{color:#535a60}.hljs-meta .hljs-string{color:#54790d}.hljs-deletion{color:#c02d2e}.hljs-addition{color:#2f6f44}.hljs-emphasis{font-style:italic}.hljs-strong{font-weight:700}
`
