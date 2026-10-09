import { browser, Browser } from 'wxt/browser';
import { AuthService } from "./auth"
import type { AuthConfig } from "./auth"
import { fetchJson, fetchStreamJson, fetchResponse, streamFromResponse, ServerSentEvent } from "./http"
import { AgentFlow } from "./flow"
import { addAgent, findAgentById, ExistingAgentError } from "./agent-repository"
import { AgentPrompt } from "../../common/src/utils/domain"
import type { StatusUpdate } from "../../common/src/components/chat/ChatMessage.vue"
import { handleToolAuthRequestsIn } from "./tool-oauth"

export abstract class AgentSource {
  abstract findAgents(authService?: AuthService): Promise<Agent[]>;

  static async loadAgentsFromUrl(url: string): Promise<Agent[]> {
    const agents: Agent[] = []
    const manifest = await Agent.findManifest(url)

    // comparing with agents-hub for backwards compatibility with environments that haven't fully migrated to tero
    if (manifest.auth && (manifest.auth.clientId === AgentType.TeroAgent || manifest.auth.clientId === "agents-hub")) {
      const authService = new AuthService(manifest.auth!)
      await authService.ensureAuthenticated()
      const tero = new TeroServer(url, manifest)
      agents.push(...await tero.findAgents(authService))
    } else {
      agents.push(new StandaloneAgent(url, manifest))
    }

    if (agents.length > 0) {
      for (const agent of agents) {
        if (await findAgentById(agent.manifest.id)) {
          throw new ExistingAgentError();
        }
        await agent.setup();
        await addAgent(agent);
      }
    }

    return agents
  }
}

const TRANSCRIPTS_CAPABILITY = 'transcripts'
const STOP_CAPABILITY = 'stop'

export abstract class Agent {
  url: string;
  logo: string;
  manifest: AgentManifest;
  activationRule: AgentRule;
  activationAction: ActivationAction;
  type: AgentType;

  protected constructor(url: string, manifest: AgentManifest, type: AgentType, icon?: string) {
    this.url = url;
    this.logo = icon || `${url}/logo.png`;
    this.manifest = manifest;
    this.activationRule = manifest.onHttpRequest?.find((r) => r.actions.find((a) => a.activate))!;
    this.activationAction = this.activationRule?.actions.find((a) => a.activate)!.activate!;
    this.type = type;
  }

  protected async postJson(url: string, body: any, authService?: AuthService): Promise<any> {
    return await fetchJson(url, await Agent.buildHttpRequest("POST", body, authService, browser.identity?.getRedirectURL()));
  }

  public static async buildHttpRequest(method: string, body?: any, authService?: AuthService, redirectUri?: string): Promise<RequestInit> {
    const headers = {} as any;
    if (body && !(body instanceof FormData)) {
      headers["Content-Type"] = "application/json";
    }
    if (authService) {
      const user = await authService.getUser();
      headers["Authorization"] = "Bearer " + user!.access_token;
    }
    if (redirectUri) {
      headers["X-Redirect-Uri"] = redirectUri;
    }
    const requestInit: RequestInit = {
      method: method,
      headers: headers,
    };
    if (body) {
      requestInit.body = body instanceof FormData ? body : JSON.stringify(body);
    }
    return requestInit;
  }

  protected async *processStreamResponse(stream: AsyncIterable<any>): AsyncIterable<MessagePart> {
    for await (const part of stream) {
      if (typeof part === "string") {
        yield { message: part };
      } else if (part && part instanceof ServerSentEvent) {
        const event = part.event || 'message'
        if (event === 'message') {
          yield { message: part.data };
        } else if (event === 'messageId') {
          yield { messageId: parseInt(part.data) };
        } else if (event === 'heartbeat') {
          continue;
        } else if (part.data) {
          yield { [`${part.event}`]: JSON.parse(part.data) };
        }
      } else if (part && typeof part === "object" && "steps" in part) {
        yield { flow: AgentFlow.fromJsonObject(part) };
      }
    }
  }

  public activatesOn(req: RequestEvent): boolean {
    return this.activationRule && this.requestMatchesRuleCondition(req, this.activationRule);
  }

  protected requestMatchesRuleCondition(req: RequestEvent, rule: AgentRule): boolean {
    return (
      new RegExp(rule.condition.urlRegex!).test(req.details.url) &&
      (!rule.condition.requestMethods || rule.condition.requestMethods!.includes(req.details.method.toLowerCase())) &&
      (!rule.condition.resourceTypes || rule.condition.resourceTypes!.includes(req.details.type)) &&
      ((!rule.condition.event && req.event === RequestEventType.OnCompleted) || rule.condition.event === req.event)
    );
  }

  public findMatchingActions(req: RequestEvent): AgentRuleAction[] {
    return this.manifest.onHttpRequest
      ? this.manifest.onHttpRequest.filter((r) => this.requestMatchesRuleCondition(req, r)).flatMap((r) => r.actions)
      : [];
  }

  public static fromJsonObject(obj: any): Agent {
    return obj.type === AgentType.TeroAgent ? TeroAgent.fromJsonObject(obj) : StandaloneAgent.fromJsonObject(obj);
  }

  public toJSON(): any {
    return { type: this.type, url: this.url, logo: this.logo, manifest: this.manifest };
  }

  public static async findManifest(url: string): Promise<AgentManifest> {
    return await fetchJson(`${url}/manifest.json`);
  }

  abstract createSession(locales: string[], authService?: AuthService): Promise<AgentSession>;

  abstract ask(msg: string, sessionId: string, parentMessageId?: number, authService?: AuthService): AsyncIterable<MessagePart>;

  abstract stopResponse(sessionId: string, authService?: AuthService): Promise<void>;

  protected abstract sessionUrl(sessionId: any): string;

  supportsTranscriptions(): boolean {
    return this.supportsCapability(TRANSCRIPTS_CAPABILITY)
  }

  private supportsCapability(capability: string): boolean {
    return this.manifest.capabilities?.includes(capability) || false
  }

  supportsStopResponse(): boolean {
    return this.supportsCapability(STOP_CAPABILITY)
  }

  abstract transcribeAudio(audioFile: Blob, sessionId: string, authService?: AuthService): Promise<string>;

  abstract solveInteractionSummary(detail: any, sessionId: string, authService?: AuthService): Promise<string>;

  abstract getPrompts(): Promise<AgentPrompt[]>;

  abstract savePrompt(p: AgentPrompt): Promise<void>;

  abstract deletePrompt(id: number): Promise<void>;

  abstract setup(): Promise<void>;

  abstract tearDown(): Promise<void>;
}

export interface MessagePart {
  messageId?: number;
  message?: string;
  flow?: AgentFlow;
  complete?: boolean;
  metadata?: {
    answerMessageId?: number
    minutesSaved?: number
    stopped?: boolean
  }
  status?: {
    action: string
    toolName?: string
    step?: string
    description?: string
    args?: string
    result?: StatusUpdate['result']
  }
}

export class StandaloneAgent extends Agent {
  private promptsRepository: LocalStoragePromptsRepository;

  constructor(url: string, manifest: AgentManifest) {
    super(url, manifest, AgentType.StandaloneAgent);
    this.promptsRepository = new LocalStoragePromptsRepository();
  }

  public static fromJsonObject(obj: any): StandaloneAgent {
    return new StandaloneAgent(obj.url, obj.manifest);
  }

  public async createSession(locales: string[], authService?: AuthService): Promise<AgentSession> {
    if (authService) {
      await authService.ensureAuthenticated();
    }
    return await this.postJson(`${this.url}/sessions`, { locales: locales }, authService);
  }

  public async *ask(msg: string, sessionId: string, parentMessageId?: number, authService?: AuthService): AsyncIterable<MessagePart> {
    const ret = await fetchStreamJson(
      `${this.sessionUrl(sessionId)}/questions`,
      await Agent.buildHttpRequest("POST", { question: msg }, authService)
    );
    yield* this.processStreamResponse(ret);
  }

  protected sessionUrl(sessionId: any): string {
    return `${this.url}/sessions/${sessionId}`;
  }

  public async stopResponse(sessionId: string, authService?: AuthService): Promise<void> {
    await this.postJson(`${this.sessionUrl(sessionId)}/stop`, undefined, authService)
  }

  public async transcribeAudio(audioFile: Blob, sessionId: string, authService?: AuthService) {
    const ret = await this.postJson(
      `${this.sessionUrl(sessionId)}/transcriptions`,
      { file: await this.blobToBase64(audioFile) },
      authService
    );
    return ret.text;
  }

  /**
   * This is the only way to convert a blob file to base64
   * We use a Promise here due to the asynchronous nature of file reading in JavaScript.
   * The FileReader API operates asynchronously, meaning the file reading process
   * doesn't complete immediately but takes some time.
   **/
  private blobToBase64(blob: Blob) {
    return new Promise((resolve, _) => {
      const reader = new FileReader();
      reader.onloadend = () => resolve(reader.result);
      reader.readAsDataURL(blob);
    });
  }

  public async solveInteractionSummary(detail: any, sessionId: string, authService?: AuthService): Promise<string> {
    const ret = await this.postJson(`${this.sessionUrl(sessionId)}/interactions`, detail, authService);
    return ret.summary;
  }

  public async savePrompt(p: AgentPrompt): Promise<void> {
    if (!p.id) {
      p.id = Math.max(...(await this.getPrompts()).map(p => p.id!)) + 1;
    }
    await this.promptsRepository.save(p, this.manifest.id);
  }

  public async getPrompts(): Promise<AgentPrompt[]> {
    return await this.promptsRepository.findByAgentId(this.manifest.id);
  }

  public async deletePrompt(id: number) {
    await this.promptsRepository.deleteByIdAndAgentId(id, this.manifest.id);
  }

  public async setup(): Promise<void> {
    if (!this.manifest.prompts) return;
    const userPrompts = await this.getPrompts();
    if (userPrompts.length === 0) {
      let promptId = 1;
      for (const p of this.manifest.prompts) {
        await this.savePrompt(new AgentPrompt(promptId++, p.name, p.content || p.text, true, true, false));
      }
    }
  }

  public async tearDown(): Promise<void> {
    this.promptsRepository.deleteByAgentId(this.manifest.id);
  }
}

export class LocalStoragePromptsRepository {

  async save(p: AgentPrompt, agentId: string): Promise<void> {
    let agents2Prompts = await this.loadLocalStorage();
    let agentPrompts = agents2Prompts[agentId] || [];
    agentPrompts = agentPrompts.filter((p2: AgentPrompt) => p2.id !== p.id);
    agentPrompts.push(p);
    agents2Prompts[agentId] = agentPrompts.sort((p1: AgentPrompt, p2: AgentPrompt) => ((p1.name || "") < (p2.name || "") ? -1 : 1));
    await this.updateLocalStorage(agents2Prompts);
  }

  private async loadLocalStorage(): Promise<Record<string, AgentPrompt[]>> {
    const { prompts } = await browser.storage.local.get("prompts");
    if (!prompts || typeof prompts !== "object" || Array.isArray(prompts)) {
      return {};
    }
    return prompts as Record<string, AgentPrompt[]>;
  }

  private async updateLocalStorage(prompts: Record<string, AgentPrompt[]>) {
    await browser.storage.local.set({ prompts: prompts });
  }

  async findByAgentId(agentId: string): Promise<AgentPrompt[]> {
    let agents2Prompts = await this.loadLocalStorage();
    return agents2Prompts[agentId] || [];
  }

  async deleteByIdAndAgentId(id: number, agentId: string): Promise<void> {
    let agents2Prompts = await this.loadLocalStorage();
    let agentPrompts = agents2Prompts[agentId] || [];
    agentPrompts = agentPrompts.filter((p: AgentPrompt) => p.id !== id);
    agents2Prompts[agentId] = agentPrompts.sort((p1: AgentPrompt, p2: AgentPrompt) => (p1.name || "") < (p2.name || "") ? -1 : 1);
    await this.updateLocalStorage(agents2Prompts);
  }

  async deleteByAgentId(agentId: string): Promise<void> {
    let agents2Prompts = await this.loadLocalStorage();
    delete agents2Prompts[agentId];
    await this.updateLocalStorage(agents2Prompts);
  }

}

export class TeroAgent extends Agent {
  private authService: AuthService;
  private shared: boolean;

  constructor(url: string, manifest: AgentManifest, icon: string, shared: boolean) {
    super(url, manifest, AgentType.TeroAgent, icon);
    this.authService = new AuthService(this.manifest.auth!);
    this.shared = shared;
  }

  public static fromJsonObject(obj: any): TeroAgent {
    return new TeroAgent(obj.url, obj.manifest, obj.logo, obj.team !== undefined);
  }

  public async createSession(locales: string[], authService?: AuthService): Promise<AgentSession> {
    const agentId = Number(this.agentIdFromSessioId(this.manifest.id));
    return await this.postJson(`${this.url}/api/threads`, { agentId: agentId }, authService);
  }

  public async *ask(msg: string, sessionId: string, parentMessageId?: number, authService?: AuthService): AsyncIterable<MessagePart> {
    const url = `${this.sessionUrl(sessionId)}/messages`;
    const formData = new FormData();
    formData.append("text", msg);
    formData.append("origin", "USER");
    if (parentMessageId) {
      formData.append("parentMessageId", parentMessageId.toString());
    }

    yield* await handleToolAuthRequestsIn(
      async () => {
        const response = await fetchResponse(url, await Agent.buildHttpRequest("POST", formData, authService));
        const stream = await streamFromResponse(response, url);
        return await this.processStreamResponse(stream);
      },
      this,
      authService
    );
  }

  protected sessionUrl(sessionId: any): string {
    return `${this.url}/api/threads/${sessionId}`;
  }

  public async stopResponse(sessionId: string, authService?: AuthService): Promise<void> {
    await this.postJson(`${this.sessionUrl(sessionId)}/stop`, undefined, authService)
  }

  public async transcribeAudio(blob: Blob, sessionId: string, authService?: AuthService): Promise<string> {
    const formData = new FormData()
    const file = new File([blob], 'audio.webm', { type: 'audio/webm' })
    formData.append('file', file)

    const data = await this.postJson(`${this.sessionUrl(sessionId)}/transcriptions`, formData, authService)
    return data.transcription
  }

  public async solveInteractionSummary(detail: any, sessionId: string, authService?: AuthService): Promise<string> {
    throw new Error("Method not implemented.");
  }

  public static fromTero(obj: any, manifest: AgentManifest, url: string): TeroAgent {
    const agentManifest: AgentManifest = {
      id: `${manifest.id}-${obj.id}`,
      name: obj.name,
      auth: manifest.auth,
      contactEmail: manifest.contactEmail,
      capabilities: [STOP_CAPABILITY, TRANSCRIPTS_CAPABILITY],
    };
    return new TeroAgent(url, agentManifest, TeroAgent.getIcon(obj), obj.team !== undefined);
  }

  public static getIcon(obj: any): string {
    return obj.icon ? `data:image/png;base64,${obj.icon}` : browser.runtime.getURL("/default-agent-icon.png");
  }

  public async getPrompts(): Promise<AgentPrompt[]> {
    const ret = await fetchJson(
      this.buildAgentPromptsUrl(),
      await Agent.buildHttpRequest("GET", undefined, this.authService)
    );
    return await ret.map((a: any) => a as AgentPrompt);
  }

  public async savePrompt(p: AgentPrompt): Promise<void> {
    if (p.id) {
      await fetchJson(this.buildAgentPromptUrl(String(p.id)), await Agent.buildHttpRequest("PUT", p, this.authService));
    } else {
      const ret = await fetchJson(this.buildAgentPromptsUrl(), await Agent.buildHttpRequest("POST", p, this.authService));
      p.id = ret.id;
    }
  }

  public async deletePrompt(id: number) {
    await fetchJson(
      this.buildAgentPromptUrl(String(id)),
      await Agent.buildHttpRequest("DELETE", undefined, this.authService)
    );
  }

  public async setup(): Promise<void> {
    // no setup needed for tero
  }

  public async tearDown(): Promise<void> {
    // no teardown needed for tero
  }

  private buildAgentPromptsUrl(): string {
    return `${this.url}/api/agents/${this.agentIdFromSessioId(this.manifest.id)}/prompts`;
  }

  private buildAgentPromptUrl(promptId: string): string {
    return `${this.buildAgentPromptsUrl()}/${promptId}`;
  }

  private agentIdFromSessioId(sessionId: string): string {
    const parts = sessionId.split("-");
    return parts[parts.length - 1];
  }
}

export class TeroServer implements AgentSource {
  constructor(private readonly serverUrl: string, private readonly manifest: AgentManifest) { }

  async findAgents(authService?: AuthService): Promise<Agent[]> {
    const ret = await fetchJson(
      `${this.serverUrl}/api/agents?pinned=true`,
      await Agent.buildHttpRequest("GET", undefined, authService)
    );
    return Promise.all(ret.map((a: any) => TeroAgent.fromTero(a, this.manifest, this.serverUrl)));
  }
}

export enum AgentType {
  TeroAgent = "tero",
  StandaloneAgent = "agent",
}

export interface AgentManifest {
  id: string
  name?: string
  capabilities?: string[]
  welcomeMessage?: string
  prompts?: ManifestPrompt[]
  onSessionClose?: EndAction
  onHttpRequest?: AgentRule[]
  pollInteractionPeriodSeconds?: number
  auth?: AuthConfig
  contactEmail: string
}

export interface ManifestPrompt {
  name: string
  content?: string
  text?: string
  starter?: boolean
}

export interface AgentRule {
  condition: AgentRuleCondition
  actions: AgentRuleAction[]
}

export interface AgentRuleCondition {
  urlRegex: string
  requestMethods?: string[]
  resourceTypes?: string[]
  event?: string
}

export interface AgentRuleAction {
  activate?: ActivationAction
  addHeader?: AddHeaderRuleAction
  recordInteraction?: RecordInteractionRuleAction
}

export interface ActivationAction {
  httpRequest?: HttpRequestAction
}

export interface EndAction {
  httpRequest: HttpRequestAction
}

export interface HttpRequestAction {
  url: string
  method?: string
}

export interface AddHeaderRuleAction {
  header: string
  value: string
}

export interface RecordInteractionRuleAction {
  httpRequest?: HttpRequestAction
}

export interface AgentSession {
  id: string
}

export class RequestEvent {
  event: RequestEventType;
  details: Browser.webRequest.OnCompletedDetails | Browser.webRequest.OnBeforeRequestDetails;

  constructor(event: RequestEventType, details: Browser.webRequest.OnCompletedDetails | Browser.webRequest.OnBeforeRequestDetails) {
    this.event = event;
    this.details = details;
  }
}

export enum RequestEventType {
  OnBeforeRequest = "onBeforeRequest",
  OnCompleted = "onCompleted",
}
