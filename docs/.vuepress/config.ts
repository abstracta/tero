import { defaultTheme } from '@vuepress/theme-default'
import { defineUserConfig } from 'vuepress'
import { viteBundler } from '@vuepress/bundler-vite'
import { getDirname, path } from 'vuepress/utils'
import { searchPlugin } from '@vuepress/plugin-search'
import { markdownIncludePlugin } from '@vuepress/plugin-markdown-include'

const __dirname = import.meta.dirname || getDirname(import.meta.url)

export default defineUserConfig({
  lang: 'en-US',
  description: 'Tero is your collaborative and secure AI agents platform that empowers software quality',
  base: '/tero/',
  head: [
    ['link', { rel: 'icon', href: '/tero/favicon.ico' }]
  ],
  theme: defaultTheme({
    logo: '/full-logo.png',
    logoDark: '/full-logo-white.png',
    editLink: false,
    lastUpdated: false,
    contributors: false,
    navbar: [
      {
        text: "Guide",
        children: [
          '/guide/introduction.md',
          '/guide/quickstart.md',
          '/guide/discover.md',
          '/guide/chat.md',
          '/guide/agents.md',
          '/guide/console.md',
          '/guide/browser-copilot.md',
          '/guide/mcp-server.md'
        ]
      },
      {
        text: '',
        link: "https://discord.gg/tP2hge7QHC"
      },
      {
        text: '',
        link: "https://github.com/abstracta/tero"
      }],
    sidebar: {
      '/guide/': [
        {
          text: 'Guide',
          children: [
            '/guide/introduction.md',
            '/guide/quickstart.md',
            '/guide/discover.md',
            '/guide/chat.md',
            '/guide/agents.md',
            '/guide/console.md',
            '/guide/browser-copilot.md',
            '/guide/mcp-server.md'
          ]
        }
      ]
    }
  }),
  alias: {
    '@theme/VPHomeFeatures.vue': path.resolve(__dirname, './components/HomeFeatures.vue')
  },
  bundler: viteBundler(),
  plugins: [
    searchPlugin({ maxSuggestions: 10 }),
    markdownIncludePlugin({ useComment: false, deep: true })
  ]
})
