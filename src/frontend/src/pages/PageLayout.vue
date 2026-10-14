<script lang="ts" setup>
import { onUnmounted } from 'vue';
import { AnimationEffect } from '../../../common/src/utils/animations';
import { useSidePanel } from '@/composables/useSidePanel';

withDefaults(defineProps<{
    leftColumnClass?: string
    centerColumnClass?: string
    rightColumnClass?: string
    animateColumns?: boolean
}>(), {
    animateColumns: true
})

const { panelComponent, panelProps, togglePanel } = useSidePanel()

onUnmounted(() => togglePanel())
</script>

<template>
    <div class="flex flex-col h-screen w-full">
        <div class="flex flex-row flex-1 p-2 sm:p-4 lg:p-6 gap-3 overflow-hidden relative">
            <Animate
                v-if="animateColumns"
                :effect="AnimationEffect.SLIDE_IN_LEFT"
                class="relative z-20"
                :class="leftColumnClass">
                <slot name="left" />
            </Animate>
            <div v-else class="relative z-20" :class="leftColumnClass">
                <slot name="left" />
            </div>
            <Animate
                v-if="animateColumns"
                :effect="AnimationEffect.SLIDE_IN_RIGHT"
                class="relative flex-1 z-20 min-w-0"
                :class="centerColumnClass">
                <slot name="center" />
            </Animate>
            <div v-else class="relative flex-1 z-20 min-w-0" :class="centerColumnClass">
                <slot name="center" />
            </div>
            <Transition name="side-panel-fade">
                <div
                    v-if="panelComponent"
                    class="absolute inset-0 z-30 bg-content/20"
                    @click="togglePanel()" />
            </Transition>
            <Transition name="side-panel-slide">
                <div
                    v-if="panelComponent"
                    class="absolute inset-y-0 right-0 z-40 h-full"
                    :class="rightColumnClass"
                    @click.stop>
                    <slot name="right">
                        <component :is="panelComponent" v-bind="panelProps" />
                    </slot>
                </div>
            </Transition>
        </div>
    </div>
</template>

<style scoped>
.side-panel-fade-enter-active,
.side-panel-fade-leave-active {
    transition: opacity 0.2s ease;
}

.side-panel-fade-enter-from,
.side-panel-fade-leave-to {
    opacity: 0;
}

.side-panel-slide-enter-active,
.side-panel-slide-leave-active {
    transition: transform 0.25s ease;
}

.side-panel-slide-enter-from,
.side-panel-slide-leave-to {
    transform: translateX(100%);
}
</style>
