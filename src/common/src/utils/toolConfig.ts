import { IconBrain, IconQuestionMark, IconWorld, IconDeviceDesktopBolt, IconDatabaseSearch, type Icon } from '@tabler/icons-vue'
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

const toolIcons: Record<string, Icon> = {
  docs: IconBrain,
  web: IconWorld,
  mcp: iconFromImage(mcpIcon),
  jira: iconFromImage(jiraIcon),
  'azure-devops': iconFromImage(azureDevopsIcon),
  zephyr: iconFromImage(zephyrIcon),
  github: iconFromImage(githubIcon),
  redmine: iconFromImage(redmineIcon),
  youtrack: iconFromImage(youtrackIcon),
  practitest: iconFromImage(practitestIcon),
  browser: IconDeviceDesktopBolt,
  sql: IconDatabaseSearch
}

export const defaultToolNames: Record<string, string> = {
  docs: 'Docs',
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
  sql: 'SQL'
}

export const mostUsedToolIds = ['docs', 'web', 'mcp', 'browser']

const wildcardToolKeys = new Set(['mcp'])

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
  return toolIcons[toolIdKey(toolId)] || IconQuestionMark
}

export const buildToolConfigName = (toolId: string): string => {
  const key = toolIdKey(toolId)
  if (key !== toolId) {
    const suffix = toolId.slice(key.length + 1)
    if (suffix && suffix !== '*') return suffix
  }
  return defaultToolNames[key] || toolId
}
