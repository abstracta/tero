import { browser } from 'wxt/browser';
import { BrowserMessage, ResizeSidebar, FlowStepExecution } from "~/utils/browser-message"
import { FlowExecutor } from "~/utils/flow"

export default defineContentScript({
  matches: ['http://*/*', 'https://*/*'],
  main(ctx) {
    let iframe = document.createElement('iframe')
    iframe.id = `browser-copilot-${browser.runtime.id}`
    setSidebarIframeStyle(iframe)
    iframe.allow = "clipboard-write; microphone"
    iframe.src = browser.runtime.getURL("/iframe.html")
    document.body.appendChild(iframe)

    // On pages that define an style for iframes with min-width set to a value and important
    // setting the inline style width gets overwritten by the page style,
    // and setting the inline style width as important makes the resulting page to don't even include the style
    // As a workaround we set the iframe an id and create a style that points to that id with important widths
    // and then updated this style whenever a resize is needed
    let iframeStyle = document.createElement('style')
    document.head.appendChild(iframeStyle)
    resize(0, iframeStyle, iframe)

    browser.runtime.onMessage.addListener(async (m: any) => {
      let msg = BrowserMessage.fromJsonObject(m)
      if (msg instanceof ResizeSidebar) {
          resize(msg.size, iframeStyle, iframe)
          msg.height && msg.position && setSidebarIframeStyle(iframe, msg.height, msg.position);
      } else if (msg instanceof FlowStepExecution) {
          return await new FlowExecutor(0).runStep(msg.step)
      }
    })
  },
});

function setSidebarIframeStyle(iframe: HTMLIFrameElement, height: string = "100%", position: "top" | "bottom" = "top") {
  let style = iframe.style
  style.height = height;
  style.position = "fixed"
  style.top = "";
  style.bottom = "";

  if (position === "bottom") {
    style.bottom = "0px";
  } else {
    style.top = "0px";
  }

  style.right = "0px"
  style.zIndex = "2147483647"
  style.border = "0px"
  style.setProperty("background-color", "transparent");
  style.setProperty("color-scheme", "auto");
}


function resize(size: number, iframeStyle: HTMLStyleElement, iframe: HTMLIFrameElement) {
    if (size > window.innerWidth) {
        size = window.innerWidth
    }
    iframeStyle.innerHTML = `#${iframe.id} {
        width: ${size}px !important;
        min-width: ${size}px !important;
    }`
}
