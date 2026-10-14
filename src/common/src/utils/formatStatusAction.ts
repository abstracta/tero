import type { ComposerTranslation } from 'vue-i18n'
import { escapeHtml } from 'markdown-it/lib/common/utils'
import type { StatusResultItem, StatusUpdate } from '../components/chat/ChatMessage.vue'

export function formatStatusAction(status: StatusUpdate, t: ComposerTranslation): string {
  const tool = escapeHtml(status.toolName ?? '')
  switch (status.action) {
    case 'statusProcessing':
      return t('statusProcessing')
    case 'preModelHook':
      return t('statusPreModel')
    case 'planning':
      return t('planning', { tool })
    case 'executingTool': {
      const params = status.args == null
        ? undefined
        : typeof status.args === 'string'
          ? status.args
          : JSON.stringify(status.args)
      const shown = params && status.argsPath ? `${params}...` : params
      const paramsSuffix = shown ? t('withParams', { params: escapeHtml(shown) }) : ''
      if (status.toolName === 'task' && status.description) {
        return t('delegatingToSubagent', { agent: escapeHtml(status.description) }) + paramsSuffix
      }
      return t('statusExecutingTool', { tool })
        + (status.step ? ' - ' + t(status.step) : '')
        + paramsSuffix
    }
    case 'executedTool':
      if (status.toolName === 'task' && status.description) {
        return t('subagentFinished', { agent: escapeHtml(status.description) })
      }
      return t('statusExecutedTool', { tool })
    case 'toolError':
      return t('toolError', { tool })
        + (status.errorId ? ' (ref: ' + escapeHtml(status.errorId) + ')' : '')
    default:
      return escapeHtml(status.action)
  }
}

export function formatStatus(status: StatusUpdate, t: ComposerTranslation): string {
  const prefix = status.agentName && status.toolName !== 'task'
    ? `<b>${escapeHtml(status.agentName)}</b>: `
    : ''
  return prefix + formatStatusAction(status, t)
}

export function resultContent(item: string | StatusResultItem): string {
  if (typeof item === 'string') return item
  return item?.content ?? ''
}

export const RESULT_PREVIEW_LENGTH = 120

export function resultPreview(item: string | StatusResultItem): string {
  const text = resultContent(item)
  return text.length >= RESULT_PREVIEW_LENGTH ? text.slice(0, RESULT_PREVIEW_LENGTH) + '...' : text
}

const stripHtml = (html: string) => html.replace(/<[^>]*>/g, '')

export function formatThoughtProcessText(statusUpdates: StatusUpdate[], t: ComposerTranslation): string {
  return statusUpdates.map((status) => {
    const lines = [stripHtml(formatStatus(status, t))]
    if (status.description?.trim() && status.toolName !== 'task') {
      lines.push(`${t('description')}: ${status.description}`)
    }
    if (typeof status.result === 'string' && status.result.trim()) {
      lines.push(`${t('result')} ${status.result}`)
    } else if (Array.isArray(status.result) && status.result.length > 0) {
      lines.push(...status.result.map((item) => `- ${resultPreview(item)}`))
    }
    return lines.join('\n')
  }).join('\n\n')
}
