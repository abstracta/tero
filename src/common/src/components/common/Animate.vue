<script lang="ts" setup>
import { ref, watch, computed, onMounted, onUnmounted } from 'vue';
import { useMotion } from '@vueuse/motion';
import { AnimationEffect, animations } from "../../utils/animations"

const props = defineProps({
  effect: {
    type: String as () => AnimationEffect,
    default: AnimationEffect.SLIDE_IN_LEFT_SPRING
  },
  // Delay based on index
  index: {
    type: Number,
    default: 0
  },
  // Base delay for each item (in ms)
  baseDelay: {
    type: Number,
    default: 80
  },
  // Whether animation should be enabled
  enabled: {
    type: Boolean,
    default: true
  },
  animateOnMount: {
    type: Boolean,
    default: true
  }
});

const target = ref<HTMLElement | null>(null);
const isVerticalExpansion = props.effect === AnimationEffect.EXPAND_VERTICAL;
const usesManualMotion = isVerticalExpansion || !props.animateOnMount;

const selectedAnimation = computed(() => {
  return animations[props.effect];
});

const enterWithDelay = computed(() => {
  const delay = Math.max(0, props.index * props.baseDelay);
  const animation = selectedAnimation.value;
  
  return {
    ...animation.enter,
    transition: {
      ...(animation.enter.transition || {}),
      delay
    }
  };
});

const variants = computed(() => {
  const animation = selectedAnimation.value;
  return {
    initial: animation.initial,
    enter: enterWithDelay.value,
    leave: 'leave' in animation ? animation.leave : animation.initial
  };
});

const { apply, set } = useMotion(target, variants.value, usesManualMotion ? {
  syncVariants: false,
  lifeCycleHooks: false,
  visibilityHooks: false,
  eventListeners: false
} : undefined);

if (isVerticalExpansion && props.animateOnMount) set('initial');

const applyEnter = () => {
  if (isVerticalExpansion && target.value) {
    const enterAnimation = apply({
      ...enterWithDelay.value,
      height: target.value.scrollHeight
    });
    void enterAnimation?.then(() => {
      if (props.enabled) set({ height: 'auto' });
    });
  } else {
    apply('enter');
  }
};

const applyLeave = () => {
  if (isVerticalExpansion && target.value) {
    set({ height: target.value.getBoundingClientRect().height });
  }
  return apply('leave');
};

let enterTimeout: ReturnType<typeof setTimeout> | undefined;

const scheduleEnter = () => {
  clearTimeout(enterTimeout);
  enterTimeout = setTimeout(() => {
    if (props.enabled) applyEnter();
  }, 20);
};

onMounted(() => {
  if (props.enabled && props.animateOnMount) {
    scheduleEnter();
  } else if (!props.enabled) {
    applyLeave();
  }
});

watch(() => props.enabled, (newValue) => {
  if (newValue) {
    scheduleEnter();
  } else {
    clearTimeout(enterTimeout);
    applyLeave();
  }
});

onUnmounted(() => clearTimeout(enterTimeout));

const reset = () => {
  const leaveAnimation = applyLeave();
  void leaveAnimation?.then(() => {
    if (props.enabled) applyEnter();
  });
};

defineExpose({
  reset
});
</script>

<template>
  <div
    ref="target"
    class="transform-gpu"
    :class="{
      'overflow-hidden': isVerticalExpansion,
      hidden: !isVerticalExpansion && !props.enabled
    }">
    <slot></slot>
  </div>
</template>
