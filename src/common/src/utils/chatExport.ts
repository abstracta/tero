import { ChatUiMessage, type StatusUpdate } from '../components/chat/ChatMessage.vue'
import { renderMarkDown } from './formatter'
import { escapeHtml } from 'markdown-it/lib/common/utils'

function formatStatusAction(status: StatusUpdate): string {
  const tool = escapeHtml(status.toolName || '')
  switch (status.action) {
    case 'statusProcessing':
      return 'Processing'
    case 'preModelHook':
      return 'Thinking'
    case 'planning':
      return 'Planning to run tools'
    case 'executingTool':
      return `Executing <b>${tool}</b>`
        + (status.step ? ` - ${escapeHtml(status.step)}` : '')
        + (status.args != null ? ` with params <b>${escapeHtml(typeof status.args === 'string' ? status.args : JSON.stringify(status.args))}</b>` : '')
    case 'executedTool':
      return `Tool <b>${tool}</b> execution finished`
    case 'toolError':
      return `Error in tool <b>${tool}</b>`
    default:
      return escapeHtml(status.action)
  }
}

function renderFileAttachments(files: { name: string }[]): string {
  if (!files || files.length === 0) return ''
  const tags = files.map(f =>
    `<span class="file-tag">${escapeHtml(f.name)}</span>`
  ).join(' ')
  return `<div class="file-attachments">${tags}</div>`
}

function renderStatusStep(status: StatusUpdate, isLast: boolean): string {
  const descriptionHtml = status.description?.trim()
    ? `<div>Description: <i>${escapeHtml(status.description)}</i></div>`
    : ''

  let resultHtml = ''
  if (Array.isArray(status.result) && status.result.length > 0) {
    resultHtml = status.result.map(doc =>
      `<div>-<b>${escapeHtml(doc)}</b></div>`
    ).join('')
  } else if (typeof status.result === 'string' && status.result.trim()) {
    resultHtml = `<div>Result: ${escapeHtml(status.result)}</div>`
  }

  return `<div class="status-step">
  <div class="status-dot-col">
    <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor"><circle cx="12" cy="12" r="5"/></svg>
    ${!isLast ? '<span class="status-line"></span>' : ''}
  </div>
  <div class="status-content">
    <div>${formatStatusAction(status)}</div>
    ${descriptionHtml}
    ${resultHtml}
  </div>
</div>`
}

function renderMessage(msg: ChatUiMessage): string {
  if (msg.isUser) {
    const textHtml = msg.text ? escapeHtml(msg.text).replace(/\n/g, '<br/>') : ''
    return `<div class="message user-message"><div class="user-bubble"><div>${textHtml}</div>${renderFileAttachments(msg.files)}</div></div>`
  }
  const thoughtHtml = msg.statusUpdates?.length
    ? `<details class="status-container"><summary class="status-header">Thought process</summary><div class="status-steps">${msg.statusUpdates.map((s, i) => renderStatusStep(s, i === msg.statusUpdates.length - 1)).join('')}</div></details>`
    : ''
  const bodyHtml = renderMarkDown(msg.text, true).replace(/<img /g, '<img onerror="this.style.display=\'none\'" ')
  return `<div class="message agent-message">${thoughtHtml}<div class="formatted-text agent-body">${bodyHtml}</div>${renderFileAttachments(msg.files)}</div>`
}

export function generateChatHtml({ messages, agentName = 'Agent', chatName = 'Chat', chartImages }: { messages: ChatUiMessage[], agentName?: string, chatName?: string, chartImages?: string[] }): string {
  const exportDate = new Date().toLocaleDateString('en-US', { year: 'numeric', month: 'long', day: 'numeric' })
  let messagesHtml = messages.map(renderMessage).join('')
  if (chartImages?.length) {
    let i = 0
    messagesHtml = messagesHtml.replace(
      /<div class="echarts"[^>]*><\/div><div class="echarts-data"[^>]*>[\s\S]*?<\/div>/g,
      () => i < chartImages.length ? `<div class="chart"><img src="${chartImages[i++]}" alt="chart"></div>` : ''
    )
  }
  return `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>${escapeHtml(chatName)} - ${escapeHtml(agentName)}</title>
<link href="https://fonts.googleapis.com/css2?family=Sora:wght@100..800&display=swap" rel="stylesheet">
<style>${CSS}</style>
</head>
<body>
<div class="chat-container">
  <header>
    <h1>${escapeHtml(agentName)}</h1>
    <div>${escapeHtml(chatName)} &middot; Exported ${exportDate}</div>
  </header>
  <div class="messages">${messagesHtml}</div>
</div>
</body>
</html>`
}

export function downloadChatHtml(html: string, filename: string): void {
  const blob = new Blob([html], { type: 'text/html;charset=utf-8' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  document.body.appendChild(a)
  a.click()
  document.body.removeChild(a)
  URL.revokeObjectURL(url)
}

const CSS = `
:root { --bg: #f4f4f4; --border: #d9d9d9; --muted: #737475; --text: #1f1f1f; --primary: #754bde; --font: 'Sora', sans-serif; --mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; }
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html { font-family: var(--font); }
body { background: #fff; color: var(--text); line-height: 1.5; }
a { color: var(--primary); text-decoration: underline; font-weight: 600; }
b, strong { font-weight: 500; }

.chat-container { max-width: 837px; margin: 0 auto; padding: 1rem; }
header { padding: 1rem 0; margin-bottom: 1rem; border-bottom: 1px solid var(--border); color: var(--muted); font-size: 0.875rem; }
header h1 { font-size: 1.25rem; font-weight: 600; color: var(--text); margin-bottom: 0.25rem; }

.messages { display: flex; flex-direction: column; gap: 0.25rem; }
.message { padding: 0.5rem; }
.user-message { display: flex; flex-direction: column; align-items: flex-end; }
.user-bubble { background: var(--bg); border-radius: 0.75rem; padding: 1rem; max-width: 75%; word-break: break-word; }
.agent-message { display: flex; flex-direction: column; gap: 0.5rem; }
.agent-body { display: flex; flex-direction: column; gap: 0.5rem; overflow-x: auto; }

.file-attachments { margin-top: 0.5rem; font-size: 0.875rem; color: var(--muted); }
.file-tag { display: inline-block; background: #fff; border: 1px solid var(--border); border-radius: 0.5rem; padding: 0.25rem 0.5rem; margin: 0.125rem; }

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

code.bg-surface-muted { background: var(--bg); padding: 0.125rem; font-size: 0.875rem; font-family: var(--mono); }

pre { margin: 0.75rem 0; }
pre > div:first-child { display: flex; align-items: center; padding: 0.5rem; background: var(--bg); border: 1px solid var(--border); border-bottom: none; border-radius: 0.5rem 0.5rem 0 0; }
pre > div:first-child > span { font-size: 0.75rem; text-transform: lowercase; color: var(--text); font-family: var(--font); }
pre > code { display: block; padding: 0.5rem; background: var(--bg); border: 1px solid var(--border); border-radius: 0 0 0.5rem 0.5rem; overflow-x: auto; font-family: var(--mono); font-size: 0.875rem; line-height: 1.5; }
pre > div:first-child > div { display: none; }

table { border-collapse: collapse; width: 100%; }
td, th { border: 1px solid var(--border); padding: 0.5rem 1rem; word-break: break-word; }
th { background: var(--bg); font-weight: 500; }
.overflow-x-auto { overflow-x: auto; }
button[onclick="downloadTableCSV(this)"] { display: none !important; }

blockquote { display: flex; gap: 0.75rem; margin: 0.5rem 0; padding: 1.5rem; border-left: 8px solid var(--bg); }

.chart { margin: 1rem 0; display: flex; justify-content: center; }
.chart svg, .chart img { max-width: 100%; height: auto; }

.status-container { margin-bottom: 0.5rem; border-bottom: 1px solid var(--border); overflow: hidden; }
.status-header { padding: 0.5rem 0; font-size: 0.875rem; font-weight: 500; color: var(--muted); cursor: pointer; user-select: none; list-style: none; display: flex; align-items: center; justify-content: space-between; }
.status-header::-webkit-details-marker { display: none; }
.status-header::after { content: '▾'; display: inline-block; margin-left: 0.5rem;}
details[open] > .status-header::after { transform: rotate(180deg); }
.status-steps { padding-top: 0.25rem; }
.status-step { display: flex; align-items: stretch; gap: 0.25rem; font-size: 0.875rem; min-width: 0; }
.status-content { min-width: 0; overflow-wrap: break-word; word-break: break-word; padding-bottom: 0.75rem; }
.status-dot-col { display: flex; flex-direction: column; align-items: center; flex-shrink: 0; color: var(--muted); }
.status-line { width: 2px; flex: 1; background: var(--border); }

.hljs{color:#2f3337;background:var(--bg)}.hljs-subst{color:#2f3337}.hljs-comment{color:#656e77}.hljs-attr,.hljs-doctag,.hljs-keyword,.hljs-meta .hljs-keyword,.hljs-section,.hljs-selector-tag{color:#015692}.hljs-attribute{color:#803378}.hljs-name,.hljs-number,.hljs-quote,.hljs-selector-id,.hljs-template-tag,.hljs-type{color:#b75501}.hljs-selector-class{color:#015692}.hljs-link,.hljs-regexp,.hljs-selector-attr,.hljs-string,.hljs-symbol,.hljs-template-variable,.hljs-variable{color:#54790d}.hljs-meta,.hljs-selector-pseudo{color:#015692}.hljs-built_in,.hljs-literal,.hljs-title{color:#b75501}.hljs-bullet,.hljs-code{color:#535a60}.hljs-meta .hljs-string{color:#54790d}.hljs-deletion{color:#c02d2e}.hljs-addition{color:#2f6f44}.hljs-emphasis{font-style:italic}.hljs-strong{font-weight:700}
`
