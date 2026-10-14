import { computed, shallowRef, type Component } from 'vue'

type PanelEntry = {
  component: Component
  props: Record<string, unknown>
}

const panelStack = shallowRef<PanelEntry[]>([])

const panelComponent = computed(() => panelStack.value[panelStack.value.length - 1]?.component ?? null)
const panelProps = computed(() => panelStack.value[panelStack.value.length - 1]?.props ?? {})
const canGoBack = computed(() => panelStack.value.length > 1)

const closePanel = () => {
  panelStack.value = []
}

export function useSidePanel() {
  const togglePanel = (component?: Component, props: Record<string, unknown> = {}) => {
    const root = panelStack.value[0]
    const isSameRoot = !!component
      && root?.component === component
      && root?.props.source === props.source

    if (!component || isSameRoot) {
      closePanel()
      return
    }

    panelStack.value = [{ component, props }]
  }

  const pushPanel = (component: Component, props: Record<string, unknown> = {}) => {
    panelStack.value = [...panelStack.value, { component, props }]
  }

  const popPanel = () => {
    if (panelStack.value.length <= 1) return
    panelStack.value = panelStack.value.slice(0, -1)
  }

  return {
    panelComponent,
    panelProps,
    canGoBack,
    togglePanel,
    pushPanel,
    popPanel,
  }
}
