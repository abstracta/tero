import checkIcon from '@tabler/icons/outline/check.svg?raw'
import { saveAs } from 'file-saver'

let isRegistered = false;

const codeExtensions: Record<string, string> = {
  bash: 'sh',
  javascript: 'js',
  kotlin: 'kt',
  markdown: 'md',
  python: 'py',
  ruby: 'rb',
  rust: 'rs',
  shell: 'sh',
  typescript: 'ts',
  zsh: 'sh',
}

const knownTextMimeTypes: string[] = ['css', 'csv', 'html', 'js', 'json', 'md']

const codeMimeTypes: Record<string, string> = {
  js: 'text/javascript;charset=utf-8;',
  md: 'text/markdown;charset=utf-8;',
  xml: 'application/xml;charset=utf-8;',
  svg: 'image/svg+xml;charset=utf-8;',
}

const getCodeExtension = (language?: string): string => {
  if (!language) return 'txt'
  const normalizedLang = language?.trim().split(/\s+/)[0].toLowerCase().replace(/[^a-z0-9]+/gi, '-')
  return codeExtensions[normalizedLang] || normalizedLang
}

const getCodeElement = (button: HTMLButtonElement): HTMLElement | null => {
  return button.closest('pre')?.querySelector<HTMLElement>('code') || null
}

const copyCode = async (code: string, button: HTMLButtonElement, t: (key: string, defaultMsg?: string) => string) => {
  try {
    const original = button.innerHTML;
    await navigator.clipboard.writeText(code);
    button.disabled = true;
    button.innerHTML = `${checkIcon}<span class="font-[sora]">${t('copiedMessage')}</span>`;
    setTimeout(() => {
      button.innerHTML = original
      button.disabled = false;
    }, 3000);
  } catch(e) {
    console.error('Copy failed', e);
  }
}

const downloadCode = (code: string, language?: string) => {
  const extension = getCodeExtension(language)
  const mimeType = codeMimeTypes[extension] || (knownTextMimeTypes.includes(extension) ? `text/${extension};charset=utf-8;` : 'text/plain;charset=utf-8;')
  saveAs(new Blob([code], { type: mimeType }), `snippet.${extension}`)
}

export function useCodeActionHandler(t: (key: string, defaultMsg?: string) => string) {
  if (isRegistered) return;

  const handler = async (e: MouseEvent) => {
    const target = e.target;
    if (!(target instanceof Element)) return;

    const codeBtn = target.closest<HTMLButtonElement>('.copy-code-btn');
    if (codeBtn) {
      const codeElement = getCodeElement(codeBtn);
      if (!codeElement) return;

      await copyCode(codeElement.innerText, codeBtn, t);
      return;
    }

    const downloadBtn = target.closest<HTMLButtonElement>('.download-code-btn');
    if (!downloadBtn) return;

    const codeElement = getCodeElement(downloadBtn);
    if (!codeElement) return;

    downloadCode(codeElement.innerText, downloadBtn.dataset.codeLang)
  };
  document.body.addEventListener('click', handler);
  isRegistered = true;
}
