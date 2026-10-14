import { IconBrain, IconQuestionMark, IconWorld, IconDeviceDesktopBolt, IconZoomCode, IconDatabaseSearch, IconBrandNotion, IconRobot, type Icon } from '@tabler/icons-vue'
import { h, type SVGAttributes } from 'vue'

import mcpIcon from '../assets/images/mcp-icon.svg'
import jiraIcon from '../assets/images/jira-icon.svg'
import redmineIcon from '../assets/images/redmine-icon.svg'
import githubIcon from '../assets/images/github-icon.svg'
import youtrackIcon from '../assets/images/youtrack-icon.svg'
import practitestIcon from '../assets/images/practitest-tool.svg'
import zephyrIcon from '../assets/images/zephyr-icon.svg'
import azureDevopsIcon from '../assets/images/azure-devops-icon.svg'

const iconFromImage = (imageSrc: string): Icon => {
  return (props: SVGAttributes) => {
    const { class: className, ...restProps } = (props ?? {}) as SVGAttributes & { class?: unknown }

    return h('img', {
      src: imageSrc,
      width: 20,
      height: 20,
      class: ['tool-menu-icon', className],
      ...restProps
    })
  }
}

const iconFromTabler = (IconComponent: Icon): Icon => {
  return (props: SVGAttributes) => {
    const { class: className, ...restProps } = (props ?? {}) as SVGAttributes & { class?: unknown }

    return h(IconComponent, {
      size: 20,
      class: ['tool-menu-icon', 'shrink-0', className],
      ...restProps
    })
  }
}

const toolIcons: Record<string, Icon> = {
  docs: iconFromTabler(IconBrain),
  codebase: iconFromTabler(IconZoomCode),
  web: iconFromTabler(IconWorld),
  mcp: iconFromImage(mcpIcon),
  jira: iconFromImage(jiraIcon),
  'azure-devops': iconFromImage(azureDevopsIcon),
  zephyr: iconFromImage(zephyrIcon),
  github: iconFromImage(githubIcon),
  redmine: iconFromImage(redmineIcon),
  youtrack: iconFromImage(youtrackIcon),
  practitest: iconFromImage(practitestIcon),
  browser: iconFromTabler(IconDeviceDesktopBolt),
  sql: iconFromTabler(IconDatabaseSearch),
  notion: iconFromTabler(IconBrandNotion),
  subagent: iconFromTabler(IconRobot)
}

export const defaultToolNames: Record<string, string> = {
  docs: 'Docs',
  codebase: 'Codebase',
  web: 'Web',
  mcp: 'MCP',
  jira: 'Jira',
  'azure-devops': 'Azure DevOps',
  zephyr: 'Zephyr',
  github: 'GitHub',
  youtrack: 'YouTrack',
  practitest: 'PractiTest',
  redmine: 'Redmine',
  browser: 'Browser',
  sql: 'SQL',
  notion: 'Notion',
  subagent: 'Subagent'
}

export const mostUsedToolIds = ['docs', 'web', 'mcp', 'browser']

const wildcardToolKeys = new Set(['mcp', 'sql', 'subagent'])

const SUBAGENT_TOOL_ID = 'subagent'

export const isSubagentToolId = (toolId: string): boolean => {
  return toolId.startsWith(SUBAGENT_TOOL_ID)
}

export const parseSubagentId = (toolId: string): number | null => {
  if (!toolId.startsWith(SUBAGENT_TOOL_ID)) {
    return null
  }
  const suffix = toolId.slice(SUBAGENT_TOOL_ID.length + 1)
  if (!/^\d+$/.test(suffix)) {
    return null
  }
  return Number(suffix)
}

export const toolIdKey = (toolId: string): string => {
  const dashIndex = toolId.indexOf('-')
  if (dashIndex === -1) {
    return toolId
  }
  const prefix = toolId.slice(0, dashIndex)
  return wildcardToolKeys.has(prefix) ? prefix : toolId
}

export const toolTranslationKey = (toolId: string): string => {
  return toolIdKey(toolId).replace(/-([a-z])/g, (_, char) => char.toUpperCase())
}

export const findToolIcon = (toolId: string): Icon => {
  return toolIcons[toolIdKey(toolId)] || iconFromTabler(IconQuestionMark)
}

export const buildToolConfigName = (toolId: string): string => {
  const key = toolIdKey(toolId)
  if (key !== toolId) {
    const suffix = toolId.slice(key.length + 1)
    if (suffix && suffix !== '*') return suffix
  }
  return defaultToolNames[key] || toolId
}
