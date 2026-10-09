export const fetchJson = async (url: string, options?: RequestInit) => {
  let ret = await fetchResponse(url, options)
  return await ret.status === 204 ? null : ret.json()
}

export const fetchResponse = async (url: string, options?: RequestInit) => {
  let ret = await fetch(url, options)

  if (ret.status < 200 || ret.status >= 300) {
    let body = await ret.text()
    console.warn(`Problem with ${options?.method ? options.method : 'GET'} ${url}`, { status: ret.status, body: body }) 
    if (ret.headers.get('Content-Type') === 'application/json') {
      let json = JSON.parse(body)
      if ('detail' in json) {
        throw new HttpServiceError(json.detail, ret.status, body)
      }
    }
    throw new HttpServiceError(undefined, ret.status, body)
  }
  return ret
}

export class HttpServiceError extends Error {
  detail?: string | object
  status?: number
  body?: string

  constructor(detail?: string | object, status?: number, body?: string) {
    super()
    this.detail = detail
    this.status = status
    this.body = body
  }

}

export async function* fetchStreamJson(url: string, options?: RequestInit): AsyncIterable<any> {
  let resp = await fetchResponse(url, options)
  yield* streamFromResponse(resp, url, options)
}

export async function* streamFromResponse(resp: Response, url: string, options?: RequestInit): AsyncIterable<any> {
  let contentType = resp.headers.get("content-type")
  if (contentType?.startsWith("text/event-stream")) {
    yield* fetchSSEStream(resp, url, options)
  } else {
    yield* await resp.json()
  }
}

async function* fetchSSEStream(resp: Response, url: string, options?: RequestInit): AsyncIterable<ServerSentEvent> {
  const reader = resp.body!.getReader()
  const decoder = new TextDecoder("utf-8")
  let leftover = ""

  while (true) {
    const { value, done } = await reader.read()
    leftover += decoder.decode(value ?? new Uint8Array(), { stream: !done })
    const parsed = ServerSentEvent.parseBuffer(leftover, done)
    leftover = parsed.leftover

    for (const event of parsed.events) {
      if (event.event === "error") {
        console.warn(`Problem while reading stream response from ${options?.method ? options.method : 'GET'} ${url}`, event)
        throw new HttpServiceError(undefined, resp.status)
      }
      if (event.event || event.data) {
        yield event
      }
    }
    if (done) break
  }
}

export class ServerSentEvent {
  event?: string
  data: string

  constructor(data: string, event?: string) {
    this.data = data
    this.event = event
  }

  public static parseBuffer(buffer: string, flush = false): { events: ServerSentEvent[], leftover: string } {
    // SSE events end on a blank line. Line endings may be CRLF, LF, or CR.
    // Hold a trailing CR (unless it already completes \r\r) so a following LF can finish CRLF.
    let work = buffer
    let suffix = ""
    if (!flush && work.endsWith("\r") && !work.endsWith("\r\r")) {
      work = work.slice(0, -1)
      suffix = "\r"
    }
    work = work.replace(/\r\n/g, "\n").replace(/\r/g, "\n")
    const parts = work.split("\n\n")
    const leftover = (parts.pop() ?? "") + suffix
    const events = parts.filter((e) => e.trim()).map((e) => ServerSentEvent.parseEvent(e))
    if (flush && leftover.trim()) {
      events.push(ServerSentEvent.parseEvent(leftover))
    }
    if (flush) {
      return { events, leftover: "" }
    }
    return { events, leftover }
  }

  private static parseEvent(event: string): ServerSentEvent {
    const parts = event.split(/\r\n|\n|\r/)
    while (parts[0] === "") parts.shift()
    let eventType
    const eventPrefix = "event: "
    if (parts[0]?.startsWith(eventPrefix)) {
      eventType = parts[0].substring(eventPrefix.length)
    }
    const dataPrefix = "data: "
    const data = parts.filter(p => p.startsWith(dataPrefix)).map(p => p.substring(dataPrefix.length)).join("\n")
    return new ServerSentEvent(data, eventType)
  }
}
