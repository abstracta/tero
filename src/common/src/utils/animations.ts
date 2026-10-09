export enum AnimationEffect {
  SLIDE_IN_LEFT = 'slideInLeft',
  SLIDE_IN_RIGHT = 'slideInRight',
  FADE_IN = 'fadeIn',
  SLIDE_IN_LEFT_SPRING = 'slideInLeftSpring',
  SLIDE_DOWN = 'slideDown',
  QUICK_SLIDE_DOWN = 'quickSlideDown',
  SLIDE_UP = 'slideUp',
  QUICK_SLIDE_UP = 'quickSlideUp',
  SLIDE_DOWN_SPRING = 'slideDownSpring',
  EXPAND_VERTICAL = 'expandVertical'
}

export const slideInLeft = {
  initial: { x: -50, opacity: 0 },
  enter: { 
    x: 0, 
    opacity: 1,
    transition: {
      type: 'spring',
      stiffness: 90,
      damping: 12,
      mass: 1.2
    }
  }
};

export const slideInRight = {
  initial: { x: 50, opacity: 0 },
  enter: { 
    x: 0, 
    opacity: 1,
    transition: {
      type: 'spring',
      stiffness: 90,
      damping: 12,
      mass: 1.2
    }
  }
};

export const fadeIn = {
  initial: { opacity: 0 },
  enter: { 
    opacity: 1,
    transition: {
      duration: 800,
      ease: 'easeOut'
    }
  }
};

export const slideInLeftSpring = {
  initial: { x: -20, opacity: 0 },
  enter: { 
    x: 0, 
    opacity: 1, 
    transition: { 
      type: 'spring', 
      stiffness: 500, 
      damping: 20 
    } 
  },
  leave: { 
    x: -20, 
    opacity: 0, 
    transition: { 
      duration: 200 
    } 
  }
};

export const slideDown = {
  initial: {
    y: -20,
    opacity: 0,
    scale: 0.95
  },
  enter: {
    y: 0,
    opacity: 1,
    scale: 1,
    transition: {
      type: 'spring',
      stiffness: 300,
      damping: 25
    }
  },
  leave: {
    y: -20,
    opacity: 0,
    scale: 0.95,
    transition: {
      duration: 200
    }
  }
};

export const quickSlideDown = {
  initial: { y: -8, opacity: 0 },
  enter: { 
    y: 0, 
    opacity: 1, 
    transition: { 
      type: 'spring', 
      stiffness: 2000, 
      damping: 25 
    } 
  },
  leave: { 
    y: -8, 
    opacity: 0, 
    transition: { 
      duration: 200 
    } 
  }
};

export const slideUp = {
  initial: {
    y: 50,
    scale: 0.9,
    opacity: 0
  },
  enter: {
    y: 0,
    scale: 1,
    opacity: 1,
    transition: {
      type: 'spring',
      stiffness: 100,
      damping: 12,
      mass: 0.9
    }
  },
  leave: {
    y: 50,
    scale: 0.9,
    opacity: 0,
    transition: {
      duration: 200
    }
  }
};

export const quickSlideUp = {
  initial: { y: 50, opacity: 0 },
  enter: { 
    y: 0, 
    opacity: 1, 
    transition: { 
      duration: 100 
    } 
  },
  leave: { 
    y: 50, 
    opacity: 0, 
    transition: { 
      duration: 100 
    } 
  }
};

export const slideDownSpring = {
  initial: { 
    y: -100,
    scale: 0.8,
    opacity: 0,
    rotateX: 15
  },
  enter: { 
    y: 0,
    scale: 1,
    opacity: 1,
    rotateX: 0,
    transition: {
      type: 'spring',
      stiffness: 200,
      damping: 20,
      mass: 1.2,
      velocity: 1
    }
  },
  leave: {
    y: -20,
    scale: 0.9,
    opacity: 0,
    transition: {
      duration: 300
    }
  }
};

export const expandVertical = {
  initial: { height: 0, opacity: 0 },
  enter: {
    height: 'auto',
    opacity: 1,
    transition: {
      height: { type: 'spring', stiffness: 300, damping: 30 },
      opacity: { duration: 250 }
    }
  },
  leave: {
    height: 0,
    opacity: 0,
    transition: {
      height: { type: 'spring', stiffness: 300, damping: 30 },
      opacity: { duration: 150 }
    }
  }
};

export const animations = {
  [AnimationEffect.SLIDE_IN_LEFT]: slideInLeft,
  [AnimationEffect.SLIDE_IN_RIGHT]: slideInRight,
  [AnimationEffect.FADE_IN]: fadeIn,
  [AnimationEffect.SLIDE_IN_LEFT_SPRING]: slideInLeftSpring,
  [AnimationEffect.SLIDE_DOWN]: slideDown,
  [AnimationEffect.QUICK_SLIDE_DOWN]: quickSlideDown,
  [AnimationEffect.SLIDE_UP]: slideUp,
  [AnimationEffect.QUICK_SLIDE_UP]: quickSlideUp,
  [AnimationEffect.SLIDE_DOWN_SPRING]: slideDownSpring,
  [AnimationEffect.EXPAND_VERTICAL]: expandVertical
};
