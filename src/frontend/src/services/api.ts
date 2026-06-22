import auth from './auth'
import moment from 'moment'
import type { JSONSchema7 } from 'json-schema'
import { UploadedFile, FileStatus, AgentPrompt } from '../../../common/src/utils/domain'
import type { StatusUpdate } from '../../../common/src/components/chat/ChatMessage.vue'

export class HttpError extends Error {
  public status: number
  public body: string
  constructor(status: number, body: string) {
    super(`Unexpected response received: ${status} - ${body}`)
    this.status = status
    this.body = body
  }
}

export class ManifestAuthConfig {
  url: string
  clientId: string
  scope: string

  constructor(url: string, clientId: string, scope: string) {
    this.url = url
    this.clientId = clientId
    this.scope = scope
  }
}

export class Manifest {
  id: string
  contactEmail: string
  auth: ManifestAuthConfig
  disablePublishGlobal: boolean

  constructor(id: string, contactEmail: string, auth: ManifestAuthConfig, disablePublishGlobal: boolean) {
    this.id = id
    this.contactEmail = contactEmail
    this.auth = auth
    this.disablePublishGlobal = disablePublishGlobal
  }
}

export const GLOBAL_TEAM_ID = 1;
export const MY_TEAM_ID = 0;
export const PRIVATE_TEAM_ID = -1;
export const PRIVATE_AGENT_ID = -1;
const PRIVATE_AGENT_ICON_BASE64 = 'iVBORw0KGgoAAAANSUhEUgAAADAAAAAwCAYAAABXAvmHAAAAAXNSR0IArs4c6QAAA+NJREFUaEPtmU2IV2UUxn9PaRFoGBkVmKZ9GFFEpQRWRh+LkDYhCUVBklGR9iW5KVoMtXGIMfowWohQaKjUrlWEWS2KgsQ2ZaGDRSGlWBJl2uk+cAYGnZn/e+feYf4T/wPDfxbvfd/z3POec57zXDHFTVPcf3oAJjuCvQj8LyMQEacB04HTE+Bx4Likf9sG3OoViogzgDnAZcBiYAEQwLfAXuAbYFDSsbaAtAYgIs4F7gZWpPOzTnLyL+AL4G3gfUm/tQGiFQARcU51ZV4C7gFmA/8A+4Ef0slLgIvzWv0KbK/WPCfpcFMQbQHoB9YAZ6bTbwAfA4fSwfOApcBDwBXA38Crkp6ddAARcRuwrXLIV+h74DFg1/B7HhF+UU5qg9gIXFoB9hVaIemjJiAaRSCrzbvAcuBP4BFJW8ZyKCJWARsAJ/xbVZSeaFKdmgKYD+wArgO+Bm6XNHRtRsQREWcBnwHXAl868SX9ON4oNAWwBNgELPRblfR0iSMRMQA8lVfuYUk7S54baU1TAMuA17PCrJPkZO5oEeHkXZ+V6nFJH3R8aJQFPQC9CEDvCtXOn4hwx51XcRsn8TrgQnfW5Dkl+z2QnfvnTGYnsUmeO3Qtq53EEXF1VT0eBPxrEP4zoAPAL4WnXwBclJRi0M5XzXBPVc02S/JvsdUCEBFXVTV/K3B5dtLigwoWmmJ/V1HveyWZdhdZXQAmaOYzHlB2Z0etHfaTPHP03Mk9P/h/86hbiryHclUiIhYBHwIzgPeS9zSmw3Y06fg7wJ3AH8AdkkwzOlpxBCLC995d18+slWRW2ZpFhFnsyznBubRuLtl8PAC8b/EBJU5kFIZeUK39JwxARHg29iTmQX63pN/HAjMswpMPICJuqBL9+WSpdsilcb2kz0cD0TUAIsJJbrq8cpisciLn4GckuXmdYt0E4JpqsH+tqlg3pYxiZ90/PgVWS3L57WoAVyaAW4F96aknN5dgJ7+bVVcDsB7kgcUqxcz01LXdw06/JOtD3QsgS6L50V3Azdk3PC9vkWTOM6J1TQ4MeZcyo0Uul+ojko5OmTI6lqOTGYH7gTezNLrGDzTRc4YDSX3JKsWL2fgelWRu1NHqdGKraZ9Uytr5qeesBX7KAzseNMYCS/FzgT7gRuCgc0eSVb6OVgwgk9OK2pMp3rpEWsA1tW5i03K+MAiDeUWSo1FkdQF4dHyhGjzuA84uOqF8kbmSZcm+0br1SFvVApBRsIh7fQ42JmxtmKXFXcBXdb8b1AaQIPycxdmhT0hNQZgrHZPkrzm1bFwAap0wwYt7ACb4BXfcvheBjq9oghdM+Qj8B4wJmEDEKTttAAAAAElFTkSuQmCC';

export enum LlmModelType {
  CHAT = 'CHAT',
  REASONING = 'REASONING'
}

export enum LlmModelVendor {
  AMAZON = 'AMAZON',
  OPENAI = 'OPENAI',
  ANTHROPIC = 'ANTHROPIC',
  GOOGLE = 'GOOGLE',
  QWEN = 'QWEN'
}


export const llmModelVendorDisplayName: Record<LlmModelVendor, string> = {
  [LlmModelVendor.OPENAI]: 'OpenAI',
  [LlmModelVendor.ANTHROPIC]: 'Anthropic',
  [LlmModelVendor.GOOGLE]: 'Google',
  [LlmModelVendor.QWEN]: 'Qwen',
  [LlmModelVendor.AMAZON]: 'Amazon'
}

export class LlmModel {
  id: string
  name: string
  description: string
  modelType: LlmModelType
  modelVendor: LlmModelVendor
  isBasic: boolean
  costMultiplier: number | null
  isBaseCostModel: boolean

  constructor(id: string, name: string, description: string, modelType: LlmModelType, modelVendor: LlmModelVendor, isBasic: boolean, costMultiplier: number | null, isBaseCostModel: boolean) {
    this.id = id
    this.name = name
    this.description = description
    this.modelType = modelType
    this.modelVendor = modelVendor
    this.isBasic = isBasic
    this.costMultiplier = costMultiplier
    this.isBaseCostModel = isBaseCostModel
  }
}

export enum LlmTemperature {
  PRECISE = 'PRECISE',
  NEUTRAL = 'NEUTRAL',
  CREATIVE = 'CREATIVE'
}

export enum ReasoningEffort {
  LOW = 'LOW',
  MEDIUM = 'MEDIUM',
  HIGH = 'HIGH'
}

export enum AgentType {
  DEEP_AGENT = 'DEEP_AGENT',
  REACT_AGENT = 'REACT_AGENT'
}

export class Evaluator {
  modelId: string
  temperature: LlmTemperature
  reasoningEffort: ReasoningEffort
  prompt: string

  constructor(modelId: string, temperature: LlmTemperature, reasoningEffort: ReasoningEffort, prompt: string) {
    this.modelId = modelId
    this.temperature = temperature
    this.reasoningEffort = reasoningEffort
    this.prompt = prompt
  }
}

export enum FileProcessor {
  BASIC = 'BASIC',
  ENHANCED = 'ENHANCED'
}

export class User {
  id?: number
  name: string
  username: string

  constructor(id: number, name: string, username: string) {
    this.id = id
    this.name = name
    this.username = username
  }
}

export class UserProfile {
  teams: TeamRole[]

  constructor(teams: TeamRole[]) {
    this.teams = teams
  }
}

export class Team {
  id: number
  name: string

  constructor(id: number, name: string) {
    this.id = id
    this.name = name
  }
}

export class TeamUser extends User {
  role: Role
  roleStatus: TeamRoleStatus
  verified: boolean

  constructor(id: number, name: string, username: string, role: Role, roleStatus: TeamRoleStatus, verified: boolean) {
    super(id, name, username)
    this.role = role
    this.roleStatus = roleStatus
    this.verified = verified
  }
}

export class TeamUpdate {
  name: string

  constructor(name: string) {
    this.name = name
  }
}

export class TeamCreate extends TeamUpdate {
  users: NewUser[]

  constructor(name: string, users: NewUser[]) {
    super(name)
    this.users = users
  }
}

export class TeamRole {
  id: number
  name: string
  role: Role
  status: TeamRoleStatus

  constructor(id: number, name: string, role: Role, status: TeamRoleStatus) {
    this.id = id
    this.name = name
    this.role = role
    this.status = status
  }
}

export enum Role {
  TEAM_OWNER = "owner",
  TEAM_MEMBER = "member",
  TEAM_EDITOR = "editor"
}

export class Agent {
  id: number
  name?: string
  description?: string
  icon?: string
  systemPrompt?: string
  modelId?: string
  temperature?: LlmTemperature
  reasoningEffort?: ReasoningEffort
  recursionLimit?: number
  activeUsers?: number
  lastUpdate?: Date
  user?: User
  canEdit?: boolean
  team?: Team
  fileProcessor?: FileProcessor
  isProtected: boolean
  agentType: AgentType

  constructor(id: number, name?: string, description?: string, icon?: string, systemPrompt?: string, modelId?: string, temperature?: LlmTemperature, recursionLimit?: number, activeUsers?: number, lastUpdate?: Date, user?: User, canEdit?: boolean, team?: Team, fileProcessor?: FileProcessor, isProtected: boolean = false, agentType: AgentType = AgentType.REACT_AGENT) {
    this.id = id
    this.name = name
    this.description = description
    this.icon = icon
    this.systemPrompt = systemPrompt
    this.modelId = modelId
    this.temperature = temperature
    this.activeUsers = activeUsers
    this.recursionLimit = recursionLimit
    this.lastUpdate = lastUpdate
    this.user = user
    this.canEdit = canEdit
    this.team = team
    this.fileProcessor = fileProcessor
    this.isProtected = isProtected
    this.agentType = agentType
  }
}

export class AgentUpdate extends Agent {
  publishPrompts?: boolean
  teamId?: number | null

  constructor(id: number, name?: string, description?: string, icon?: string, systemPrompt?: string, modelId?: string, temperature?: LlmTemperature, recursionLimit?: number, activeUsers?: number, lastUpdate?: Date, user?: User, canEdit?: boolean, publishPrompts?: boolean, teamId?: number | null, fileProcessor?: FileProcessor, isProtected?: boolean, agentType?: AgentType) {
    super(id, name, description, icon, systemPrompt, modelId, temperature, recursionLimit, activeUsers, lastUpdate, user, canEdit, undefined, fileProcessor, isProtected, agentType)
    this.publishPrompts = publishPrompts
    this.teamId = teamId
  }
}

export enum AutomaticAgentField {
  NAME = 'NAME',
  DESCRIPTION = 'DESCRIPTION',
  SYSTEM_PROMPT = 'SYSTEM_PROMPT'
}

export interface AgentPromptUpdate {
  name?: string
  content?: string
  shared?: boolean
}

export class AgentTool {
  id: string
  name: string
  description: string
  configSchema: JSONSchema7

  constructor(id: string, name: string, description: string, configSchema: JSONSchema7) {
    this.id = id
    this.name = name
    this.description = description
    this.configSchema = configSchema
  }
}

export class AgentToolConfig {
  toolId: string
  config: Record<string, any>

  constructor(toolId: string, config: Record<string, any>) {
    this.toolId = toolId
    this.config = config
  }
}

export class DocToolFile {
  id: number
  name: string
  status: FileStatus
  agentId: number
  description: string
  fileProcessor: FileProcessor
  processedContent: string

  constructor(id: number, name: string, status: FileStatus, agentId: number, description: string, fileProcessor: FileProcessor, processedContent: string) {
    this.id = id
    this.name = name
    this.status = status
    this.agentId = agentId
    this.description = description
    this.fileProcessor = fileProcessor
    this.processedContent = processedContent
  }
}

export class Thread {
  id: number
  name: string
  agent: Agent
  creation: Date
  lastMessage: Date

  constructor(id: number, name: string, agent: Agent, creation: Date) {
    this.id = id
    this.name = name
    this.agent = agent
    this.creation = creation
    this.lastMessage = new Date()
  }
}

export enum ThreadMessageOrigin {
  USER = 'USER',
  AGENT = 'AGENT',
  SYSTEM = 'SYSTEM'
}

export class ThreadMessage {
  id: number
  text: string
  timestamp: Date
  origin: ThreadMessageOrigin
  children: ThreadMessage[]
  minutesSaved?: number
  feedbackText?: string
  hasPositiveFeedback?: boolean
  files?: UploadedFile[]
  stopped?: boolean
  statusUpdates: StatusUpdate[] = []

  constructor(id: number, text: string, timestamp: Date, origin: ThreadMessageOrigin, children: ThreadMessage[], minutesSaved?: number, feedbackText?: string, hasPositiveFeedback?: boolean, stopped?: boolean, statusUpdates?: StatusUpdate[]) {
    this.id = id
    this.text = text
    this.timestamp = timestamp
    this.origin = origin
    this.children = children
    this.minutesSaved = minutesSaved
    this.feedbackText = feedbackText
    this.hasPositiveFeedback = hasPositiveFeedback
    this.stopped = stopped
    this.statusUpdates = statusUpdates || []
  }
}

export interface ThreadMessageUpdate {
  minutesSaved?: number
  feedbackText?: string
  hasPositiveFeedback?: boolean
}

export class ThreadMessageFile {
  processedContent?: string
  status: FileStatus

  constructor(status: FileStatus, processedContent?: string) {
    this.status = status
    this.processedContent = processedContent
  }
}

export class BudgetUsage {
  usagePercent: number

  constructor(usagePercent: number) {
    this.usagePercent = usagePercent
  }
}

export class AgentItem {
  agentId: number
  agentName: string
  team: Team | null
  icon?: string
  authorName?: string
  activeUsers: number
  previousActiveUsers: number

  constructor(agentId: number, agentName: string, icon: string, team: Team | null, authorName: string, activeUsers: number, previousActiveUsers: number) {
    this.agentId = agentId
    this.agentName = agentName
    this.team = team
    this.icon = icon
    this.authorName = authorName
    this.activeUsers = activeUsers
    this.previousActiveUsers = previousActiveUsers
  }
}

export class ImpactSummary {
  humanHours: number
  aiHours: number
  previousHumanHours: number
  previousAiHours: number

  constructor(humanHours: number, aiHours: number, previousHumanHours: number = 0, previousAiHours: number = 0) {
    this.humanHours = humanHours
    this.aiHours = aiHours
    this.previousHumanHours = previousHumanHours
    this.previousAiHours = previousAiHours
  }
}

export class AgentImpactItem extends AgentItem {
  minutesSaved: number
  previousMinutesSaved: number
  isExternalAgent?: boolean

  constructor(agentId: number, agentName: string, team: Team | null, activeUsers: number, minutesSaved: number, previousActiveUsers: number = 0, previousMinutesSaved: number = 0, icon: string, authorName: string, isExternalAgent?: boolean) {
    super(agentId, agentName, icon, team, authorName, activeUsers, previousActiveUsers)
    this.minutesSaved = minutesSaved
    this.previousMinutesSaved = previousMinutesSaved
    this.isExternalAgent = isExternalAgent
  }
}

export class UserImpactItem {
  userId: number
  userName: string
  minutesSaved: number
  monthlyHours: number
  previousMinutesSaved: number

  constructor(userId: number, userName: string, minutesSaved: number, monthlyHours: number, previousMinutesSaved: number = 0) {
    this.userId = userId
    this.userName = userName
    this.minutesSaved = minutesSaved
    this.monthlyHours = monthlyHours
    this.previousMinutesSaved = previousMinutesSaved
  }
}

export class UsageSummary {
  activeUsers: number
  totalThreads: number
  previousActiveUsers: number
  previousTotalThreads: number

  constructor(activeUsers: number, totalThreads: number, previousActiveUsers: number, previousTotalThreads: number) {
    this.activeUsers = activeUsers
    this.totalThreads = totalThreads
    this.previousActiveUsers = previousActiveUsers
    this.previousTotalThreads = previousTotalThreads
  }
}

export class AgentUsageItem extends AgentItem {
  totalThreads: number
  previousTotalThreads: number

  constructor(agentId: number, agentName: string, team: Team | null, totalThreads: number, previousTotalThreads: number, icon: string, authorName: string, activeUsers: number, previousActiveUsers: number) {
    super(agentId, agentName, icon, team, authorName, activeUsers, previousActiveUsers)
    this.totalThreads = totalThreads
    this.previousTotalThreads = previousTotalThreads
  }
}

export class UserUsageItem {
  userId: number
  userName: string
  totalThreads: number
  previousTotalThreads: number

  constructor(userId: number, userName: string, totalThreads: number, previousTotalThreads: number) {
    this.userId = userId
    this.userName = userName
    this.totalThreads = totalThreads
    this.previousTotalThreads = previousTotalThreads
  }
}

export class NewUser {
  username: string;
  role: Role;

  constructor(username: string, role: Role) {
    this.username = username;
    this.role = role;
  }
}

export enum TeamRoleStatus {
  ACCEPTED = 'accepted',
  PENDING = 'pending',
  REJECTED = 'rejected'
}

export class ExternalAgent {
  id: number
  name: string
  icon: string

  constructor(id: number, name: string, icon: string) {
    this.id = id
    this.name = name
    this.icon = icon
  }
}

export class NewExternalAgent {
  name: string

  constructor(name: string) {
    this.name = name
  }
}

export class ExternalAgentTimeSaving {
  id: number
  date: Date
  minutesSaved: number

  constructor(id: number, date: Date, minutesSaved: number) {
    this.id = id
    this.date = date
    this.minutesSaved = minutesSaved
  }
}

export class TestCase {
  agentId: number
  lastUpdate: Date
  thread: Thread

  constructor(agentId: number, lastUpdate: Date, thread: Thread) {
    this.agentId = agentId
    this.lastUpdate = lastUpdate
    this.thread = thread
  }
}

export enum TestSuiteRunStatus {
  RUNNING = 'RUNNING',
  SUCCESS = 'SUCCESS',
  FAILURE = 'FAILURE',
  CANCELLING = 'CANCELLING'
}

export class TestSuiteRun {
  id: number
  agentId: number
  status: TestSuiteRunStatus
  executedAt: Date
  completedAt?: Date
  totalTests: number
  passedTests: number
  failedTests: number
  errorTests: number
  skippedTests: number

  constructor(
    id: number,
    agentId: number,
    status: TestSuiteRunStatus,
    executedAt: Date,
    totalTests: number,
    passedTests: number,
    failedTests: number,
    errorTests: number,
    skippedTests: number,
    completedAt?: Date,
  ) {
    this.id = id
    this.agentId = agentId
    this.status = status
    this.executedAt = executedAt
    this.totalTests = totalTests
    this.passedTests = passedTests
    this.failedTests = failedTests
    this.errorTests = errorTests
    this.skippedTests = skippedTests
    this.completedAt = completedAt
  }
}

export class TestCaseResult {
  id?: number
  testCaseId: number
  testSuiteRunId?: number
  executedAt: Date
  status: TestCaseResultStatus
  testCaseName: string
  evaluatorAnalysis?: string
  errorCode?: string

  constructor(testCaseId: number, executedAt: Date, status: TestCaseResultStatus, testCaseName: string, id?: number, testSuiteRunId?: number, evaluatorAnalysis?: string, errorCode?: string) {
    this.testCaseId = testCaseId
    this.executedAt = executedAt
    this.status = status
    this.testCaseName = testCaseName
    this.testSuiteRunId = testSuiteRunId
    this.id = id
    this.evaluatorAnalysis = evaluatorAnalysis
    this.errorCode = errorCode
  }
}

export enum TestCaseResultStatus {
  PENDING = 'PENDING',
  RUNNING = 'RUNNING',
  SUCCESS = 'SUCCESS',
  FAILURE = 'FAILURE',
  ERROR = 'ERROR',
  SKIPPED = 'SKIPPED'
}

export type TestSuiteExecutionStreamEvent =
  | { type: 'suite.test.start'; data: { testCaseId: number; resultId: number } }
  | { type: 'suite.test.metadata'; data: { testCaseId: number; resultId: number } }
  | { type: 'suite.test.phase'; data: { phase: string; status?: string; evaluation?: any } }
  | { type: 'suite.test.userMessage'; data: { id: number; text: string } }
  | { type: 'suite.test.agentMessage.start'; data: { id: number } }
  | { type: 'suite.test.agentMessage.chunk'; data: { id: number; chunk: string } }
  | { type: 'suite.test.agentMessage.complete'; data: { id: number; text: string } }
  | { type: 'suite.test.executionStatus'; data: any }
  | { type: 'suite.test.error'; data: { message: string } }
  | { type: 'suite.test.complete'; data: { testCaseId: number; resultId: number; status: string; errorCode?: string; evaluation?: any } }
  | { type: 'suite.complete'; data: { suiteRunId: number; status: string; totalTests: number; passed: number; failed: number; errors: number; skipped: number } }
  | { type: 'suite.error'; data: {} }

export class TestCaseNewThreadMessage {
  text: string
  origin: ThreadMessageOrigin

  constructor(text: string, origin: ThreadMessageOrigin) {
    this.text = text
    this.origin = origin
  }
}

export class TestCaseThreadMessageUpdate {
  text: string

  constructor(text: string) {
    this.text = text
  }
}

let config: Manifest | null = null

export const findManifest = async (): Promise<Manifest> => {
  if (!config) {
    config = await new ApiService().findManifest()
  }
  return config
}

export interface AgentImportResult {
  unavailableTools: string[]
  toolsRequiringAuthentication: string[]
  unavailableModel?: string
  defaultModel?: string
}

export class ApiService {

  async findManifest(): Promise<Manifest> {
    const response = await fetch(`/manifest.json`)
    return await response.json()
  }

  private async fetchJson(path: string, method: string = 'GET', body: object | undefined = undefined, authEnabled: boolean = true) {
    const response = await this.fetch(path, method, body, authEnabled)
    return await response.json()
  }

  private async fetch(path: string, method: string = 'GET', body: object | undefined = undefined, authEnabled: boolean = true) {
    const headers: Record<string, string> = {}
    if (authEnabled) {
      headers['Authorization'] = await this.getUserAuth()
    }
    if (body && !(body instanceof FormData)) {
      headers['Content-Type'] = 'application/json'
    }
    const response = await fetch(`/api${path}`, {
      method: method,
      headers: headers,
      body: body ? (body instanceof FormData ? body : JSON.stringify(body)) : undefined
    })
    await this.checkResponse(response)
    return response
  }

  private cleanSearchParams(obj: Record<string, string | number | boolean | Date | null | undefined>): string {
    const clean = Object.fromEntries(
      Object.entries(obj)
        .filter(([_, v]) => v !== undefined && v !== null)
        .map(([k, v]) => [k, v instanceof Date ? v.toISOString() : v])
    )
    return new URLSearchParams(clean as Record<string, string>).toString()
  }

  private async getUserAuth(): Promise<string> {
    const user = await auth.getUser()
    return `Bearer ${user!.access_token}`
  }

  private async checkResponse(response: Response) {
    if (!response.ok) {
      const body = await response.text()
      throw new HttpError(response.status, body)
    }
  }

  async findUserAgents(text: string = ''): Promise<Agent[]> {
    return await this.fetchJson(`/agents?pinned=true&text=${encodeURIComponent(text)}`)
  }

  async findOwnAgents(limit: number, offset: number = 0): Promise<Agent[]> {
    return await this.fetchJson(`/agents?own=true&limit=${limit}&offset=${offset}`)
  }

  async findTopUsedAgents(limit: number, offset: number = 0, teamId: number | null = null): Promise<Agent[]> {
    const params = new URLSearchParams({ sort: 'ACTIVE_USERS', limit: limit.toString(), offset: offset.toString() })
    if (teamId) params.set('team_id', teamId.toString())
    return await this.fetchJson(`/agents?${params.toString()}`)
  }

  async findNewestAgents(limit: number, offset: number = 0, teamId: number | null = null): Promise<Agent[]> {
    const params = new URLSearchParams({ sort: 'LAST_UPDATE', limit: limit.toString(), offset: offset.toString() })
    if (teamId) params.set('team_id', teamId.toString())
    return await this.fetchJson(`/agents?${params.toString()}`)
  }

  async findAgentsByText(text: string, limit: number, offset: number = 0): Promise<Agent[]> {
    return await this.fetchJson(`/agents?text=${encodeURIComponent(text)}&limit=${limit}&offset=${offset}`)
  }

  async findDefaultAgent(): Promise<Agent> {
    return await this.fetchJson(`/agents/default`)
  }

  async addUserAgent(agentId: number) {
    await this.post(`/agents/${agentId}/pin`)
  }

  private async post(path: string, body: object | undefined = undefined) {
    return await this.fetchJson(path, 'POST', body)
  }

  async removeUserAgent(agentId: number) {
    await this.delete(`/agents/${agentId}/pin`)
  }

  private async delete(path: string) {
    await this.fetch(path, 'DELETE')
  }

  async newAgent(): Promise<Agent> {
    return await this.post('/agents')
  }

  async cloneAgent(agentId: number): Promise<Agent> {
    return await this.post(`/agents/${agentId}/clone`)
  }

  async exportAgent(agentId: number): Promise<File> {
    return await this.downloadFile(`/agents/${agentId}/dist`)
  }

  async importAgent(agentId: number, file: File): Promise<AgentImportResult> {
    const formData = new FormData()
    formData.append('file', file)
    return await this.put(`/agents/${agentId}/dist`, formData)
  }

  async findAgentById(agentId: number): Promise<Agent> {
    return await this.fetchJson(`/agents/${agentId}`)
  }

  async updateAgent(agent: AgentUpdate): Promise<Agent> {
    return await this.put(`/agents/${agent.id}`, agent)
  }

  private async put(path: string, body: object | undefined = undefined) {
    return await this.fetchJson(path, 'PUT', body)
  }

  async generateAgentField(agentId: number, field: AutomaticAgentField): Promise<string> {
    return await this.post(`/agents/${agentId}/fields/${field}`)
  }

  async findModels(): Promise<LlmModel[]> {
    return await this.fetchJson('/models')
  }

  async findAgentTools(): Promise<AgentTool[]> {
    return await this.fetchJson(`/tools`)
  }

  async findAgentToolConfigs(agentId: number): Promise<AgentToolConfig[]> {
    return await this.fetchJson(`/agents/${agentId}/tools`)
  }

  async configureAgentTool(agentId: number, config: AgentToolConfig): Promise<AgentToolConfig> {
    return await this.post(`/agents/${agentId}/tools`, config)
  }

  async removeAgentToolConfig(agentId: number, toolId: string) {
    await this.delete(`/agents/${agentId}/tools/${toolId}`)
  }

  async findAgentToolFiles(agentId: number, toolId: string): Promise<UploadedFile[]> {
    return await this.fetchJson(`/agents/${agentId}/tools/${toolId}/files`)
  }

  async findAgentToolFile(agentId: number, toolId: string, fileId: number): Promise<DocToolFile> {
    return await this.fetchJson(`/agents/${agentId}/tools/${toolId}/files/${fileId}`)
  }

  async uploadAgentToolFile(agentId: number, toolId: string, file: File): Promise<UploadedFile> {
    const formData = new FormData()
    formData.append('file', file)
    return await this.post(`/agents/${agentId}/tools/${toolId}/files`, formData)
  }

  async downloadAgentToolFile(agentId: number, toolId: string, fileId: number): Promise<File> {
    return await this.downloadFile(`/agents/${agentId}/tools/${toolId}/files/${fileId}/content`)
  }

  private async downloadFile(url: string): Promise<File> {
    const response = await this.fetch(url)
    let filename = response.headers.get('content-disposition')?.match(/filename="(.*)"/)?.[1] || 'downloaded-file'
    filename = decodeURIComponent(filename)
    const blob = await response.blob()
    const contentType = response.headers.get('content-type') || blob.type || 'application/octet-stream'
    return new File([blob], filename, { type: contentType })
  }

  async updateAgentToolFile(agentId: number, toolId: string, fileId: number, file: File) {
    const formData = new FormData()
    formData.append('file', file)
    await this.put(`/agents/${agentId}/tools/${toolId}/files/${fileId}`, formData)
  }

  async deleteAgentToolFile(agentId: number, toolId: string, fileId: number) {
    await this.delete(`/agents/${agentId}/tools/${toolId}/files/${fileId}`)
  }

  async findAgentEvaluator(agentId: number): Promise<Evaluator> {
    return await this.fetchJson(`/agents/${agentId}/evaluator`)
  }

  async saveAgentEvaluator(agentId: number, evaluator: Evaluator): Promise<Evaluator> {
    return await this.put(`/agents/${agentId}/evaluator`, evaluator)
  }

  async findAgentPrompts(agentId: number): Promise<AgentPrompt[]> {
    return await this.fetchJson(`/agents/${agentId}/prompts`)
  }

  async newAgentPrompt(agentId: number, newAgentPrompt: AgentPrompt): Promise<AgentPrompt> {
    return await this.post(`/agents/${agentId}/prompts`, newAgentPrompt)
  }

  async updateAgentPrompt(agentId: number, promptId: number, prompt: AgentPromptUpdate): Promise<AgentPrompt> {
    return await this.fetchJson(`/agents/${agentId}/prompts/${promptId}`, 'PUT', prompt)
  }

  async deleteAgentPrompt(agentId: number, promptId: number) {
    await this.delete(`/agents/${agentId}/prompts/${promptId}`)
  }

  async findTestCases(agentId: number): Promise<TestCase[]> {
    return await this.fetchJson(`/agents/${agentId}/tests`)
  }

  async findTestCase(agentId: number, testCaseId: number): Promise<TestCase> {
    return await this.fetchJson(`/agents/${agentId}/tests/${testCaseId}`)
  }

  async addTestCase(agentId: number, fromThreadId?: number, branchMessageId?: number): Promise<TestCase> {
    const requestBody: { fromThreadId?: number; branchMessageId?: number } = {}
    if (fromThreadId !== undefined) {
      requestBody.fromThreadId = fromThreadId
    }
    if (branchMessageId !== undefined) {
      requestBody.branchMessageId = branchMessageId
    }
    return await this.post(`/agents/${agentId}/tests`, Object.keys(requestBody).length > 0 ? requestBody : undefined)
  }

  async updateTestCase(agentId: number, testCaseId: number, name: string): Promise<TestCase> {
    return await this.put(`/agents/${agentId}/tests/${testCaseId}`, { name })
  }

  async cloneTestCase(agentId: number, testCaseId: number): Promise<TestCase> {
    return await this.post(`/agents/${agentId}/tests/${testCaseId}/clone`)
  }

  async deleteTestCase(agentId: number, testCaseId: number) {
    await this.delete(`/agents/${agentId}/tests/${testCaseId}`)
  }

  async findTestCaseMessages(agentId: number, testCaseId: number): Promise<ThreadMessage[]> {
    return await this.fetchJson(`/agents/${agentId}/tests/${testCaseId}/messages`)
  }

  async findTestCaseEvaluator(agentId: number, testCaseId: number): Promise<Evaluator> {
    return await this.fetchJson(`/agents/${agentId}/tests/${testCaseId}/evaluator`)
  }

  async saveTestCaseEvaluator(agentId: number, testCaseId: number, config: Evaluator): Promise<Evaluator> {
    return await this.put(`/agents/${agentId}/tests/${testCaseId}/evaluator`, config)
  }

  async addTestCaseMessage(agentId: number, testCaseId: number, message: TestCaseNewThreadMessage): Promise<ThreadMessage> {
    return await this.fetchJson(`/agents/${agentId}/tests/${testCaseId}/messages`, 'POST', message)
  }

  async updateTestCaseMessage(agentId: number, testCaseId: number, messageId: number, message: TestCaseThreadMessageUpdate): Promise<ThreadMessage> {
    return await this.fetchJson(`/agents/${agentId}/tests/${testCaseId}/messages/${messageId}`, 'PUT', message)
  }

  async deleteTestCaseMessageAndFollowing(agentId: number, testCaseId: number, messageId: number): Promise<void> {
    await this.delete(`/agents/${agentId}/tests/${testCaseId}/messages/${messageId}`)
  }

  async runTestSuite(agentId: number, testCaseIds?: number[]): Promise<TestSuiteRun> {
    const requestBody = testCaseIds ? { test_case_ids: testCaseIds } : {};
    const suiteRun = await this.fetchJson(`/agents/${agentId}/tests/runs`, 'POST', requestBody);
    return this.parseTestSuiteRunDates(suiteRun);
  }

  async *streamTestSuiteUpdates(agentId: number, suiteRunId: number): AsyncIterable<TestSuiteExecutionStreamEvent> {
    const url = `/agents/${agentId}/tests/runs/${suiteRunId}/stream`;
    const resp = await this.fetch(url, 'GET');

    const contentType = resp.headers.get('content-type')
    if (contentType?.startsWith('text/event-stream')) {
      const stream = await this.fetchSSEStream(resp, url)
      for await (const part of stream) {
        yield { type: part.event as TestSuiteExecutionStreamEvent['type'], data: part.data };
      }
    }
  }

  async stopTestSuiteRun(agentId: number, suiteRunId: number): Promise<void> {
    await this.post(`/agents/${agentId}/tests/runs/${suiteRunId}/stop`)
  }

  async findTestSuiteRuns(agentId: number, limit: number = 20, offset: number = 0): Promise<TestSuiteRun[]> {
    const searchParams = new URLSearchParams({ limit: limit.toString(), offset: offset.toString() });
    const suiteRuns = await this.fetchJson(`/agents/${agentId}/tests/runs?${searchParams.toString()}`)
    return suiteRuns.map((suiteRun: TestSuiteRun) => this.parseTestSuiteRunDates(suiteRun))
  }

  private parseTestSuiteRunDates(suiteRun: TestSuiteRun): TestSuiteRun {
    return {
      ...suiteRun,
      executedAt: moment.utc(suiteRun.executedAt).toDate(),
      completedAt: suiteRun.completedAt ? moment.utc(suiteRun.completedAt).toDate() : undefined
    }
  }

  async findTestSuiteRunResults(agentId: number, suiteRunId: number): Promise<TestCaseResult[]> {
    return await this.fetchJson(`/agents/${agentId}/tests/runs/${suiteRunId}/results`)
  }

  async deleteTestSuiteRun(agentId: number, suiteRunId: number): Promise<void> {
    await this.delete(`/agents/${agentId}/tests/runs/${suiteRunId}`)
  }

  async findTestSuiteRunResultMessages(agentId: number, suiteRunId: number, resultId: number): Promise<ThreadMessage[]> {
    return await this.fetchJson(`/agents/${agentId}/tests/runs/${suiteRunId}/results/${resultId}/messages`)
  }

  async findLastChats(limit: number, excludeEmpty: boolean = false): Promise<Thread[]> {
    return await this.fetchJson(`/threads?limit=${limit}&exclude_empty=${excludeEmpty}`)
  }

  async findChatsByText(text: string, limit: number, excludeEmpty: boolean = false, agentId?: number): Promise<Thread[]> {
    return await this.fetchJson(`/threads?text=${encodeURIComponent(text)}&limit=${limit}&exclude_empty=${excludeEmpty}` + (agentId ? `&agent_id=${agentId}` : ''))
  }

  async startThread(agentId: number): Promise<Thread> {
    return await this.post(`/threads`, { agentId: agentId })
  }

  async findThread(threadId: number): Promise<Thread> {
    return await this.fetchJson(`/threads/${threadId}`)
  }

  async findThreadMessages(threadId: number): Promise<ThreadMessage[]> {
    return await this.fetchJson(`/threads/${threadId}/messages`)
  }

  async findThreadMessageFile(threadId: number, fileId: number): Promise<ThreadMessageFile> {
    return await this.fetchJson(`/threads/${threadId}/files/${fileId}`)
  }

  async downloadThreadMessageFile(threadId: number, fileId: number): Promise<File> {
    return await this.downloadFile(`/threads/${threadId}/files/${fileId}/content`)
  }

  async updateThreadMessage(threadId: number, threadMessageId: number, updatedThreadMessage: ThreadMessageUpdate): Promise<ThreadMessage> {
    return await this.fetchJson(`/threads/${threadId}/messages/${threadMessageId}`, 'PUT', updatedThreadMessage)
  }

  async updateThread(thread: Thread): Promise<Thread> {
    return await this.fetchJson(`/threads/${thread.id}`, 'PUT', thread)
  }

  async deleteThread(threadId: number) {
    await this.delete(`/threads/${threadId}`)
  }

  async getImpactSummary(fromDate: Date, toDate: Date, teamId: number): Promise<ImpactSummary> {
    const params = this.cleanSearchParams({ from_date: fromDate, to_date: toDate, team_id: teamId })
    return await this.fetchJson(`/impact/summary?${params}`)
  }

  async getImpactTopAgents(privateAgentsName: string, fromDate: Date, toDate: Date, teamId: number, search?: string, limit?: number, offset?: number, userId?: number): Promise<AgentImpactItem[]> {
    const params = this.cleanSearchParams({ from_date: fromDate, to_date: toDate, team_id: teamId, search: search, limit: limit, offset: offset, user_id: userId })
    const agents = await this.fetchJson(`/impact/agents?${params}`)

    return agents.map((agent: AgentImpactItem) => {
      if (agent.agentId === PRIVATE_AGENT_ID) {
        agent.agentName = privateAgentsName
        agent.icon = PRIVATE_AGENT_ICON_BASE64
        agent.team = {
          id: PRIVATE_TEAM_ID,
          name: privateAgentsName
        }
      }
      return agent
    })
  }

  async getImpactTopUsers(fromDate: Date, toDate: Date, teamId: number, search?: string, limit?: number, offset?: number, agentId?: number, isExternalAgent?: boolean): Promise<UserImpactItem[]> {
    const params = this.cleanSearchParams({ from_date: fromDate, to_date: toDate, team_id: teamId, search: search, limit: limit, offset: offset, agent_id: agentId, is_external_agent: isExternalAgent })
    return await this.fetchJson(`/impact/users?${params}`)
  }

  async getUsageSummary(fromDate: Date, toDate: Date, teamId: number): Promise<UsageSummary> {
    const params = this.cleanSearchParams({ from_date: fromDate, to_date: toDate, team_id: teamId })
    return await this.fetchJson(`/usage/summary?${params}`)
  }

  async getUsageTopAgents(privateAgentsName: string, fromDate: Date, toDate: Date, teamId: number, search?: string, limit?: number, offset?: number, userId?: number): Promise<AgentUsageItem[]> {
    const params = this.cleanSearchParams({ from_date: fromDate, to_date: toDate, team_id: teamId, search: search, limit: limit, offset: offset, user_id: userId })
    const agents = await this.fetchJson(`/usage/agents?${params}`)

    return agents.map((agent: AgentUsageItem) => {
      if (agent.agentId === PRIVATE_AGENT_ID) {
        agent.agentName = privateAgentsName
        agent.icon = PRIVATE_AGENT_ICON_BASE64
        agent.team = {
          id: PRIVATE_TEAM_ID,
          name: privateAgentsName
        }
      }
      return agent
    })
  }

  async getUsageTopUsers(fromDate: Date, toDate: Date, teamId: number, search?: string, limit?: number, offset?: number, agentId?: number): Promise<UserUsageItem[]> {
    const params = this.cleanSearchParams({ from_date: fromDate, to_date: toDate, team_id: teamId, search: search, limit: limit, offset: offset, agent_id: agentId })
    return await this.fetchJson(`/usage/users?${params}`)
  }

  async stopMessageResponse(threadId: number) {
    await this.post(`/threads/${threadId}/stop`)
  }

  async *sendMessage(threadId: number, text: string, files?: UploadedFile[], parentMessageId?: number, isInAgentEdition?: boolean): AsyncIterable<ThreadMessagePart> {
    const url = `/threads/${threadId}/messages`

    const formData = new FormData()
    formData.append('text', text)
    formData.append('origin', 'USER')
    if (parentMessageId != null) formData.append('parentMessageId', parentMessageId.toString())
    if (isInAgentEdition != null) formData.append('isInAgentEdition', isInAgentEdition.toString())
    if (files) {
      const fileIds = []
      for (const file of files) {
        if (file.file) {
          this.addFileToForm(file.file, formData, 'files')
        } else {
          fileIds.push(file.id)
        }
      }
      formData.append('fileIds', fileIds.join(','))
    }
    const resp = await this.fetch(url, 'POST', formData, true)

    const contentType = resp.headers.get('content-type')
    if (contentType?.startsWith('text/event-stream')) {
      const ret = await this.fetchSSEStream(resp, url)
      for await (const part of ret) {
        if (part.event == 'userMessage') {
          yield { userMessage: part.data }
        } else if (part.event == 'message') {
          yield { answerText: part.data }
        } else if (part.event == 'metadata') {
          yield {
            metadata: {
              answerMessageId: part.data.answerMessageId,
              files: part.data.files,
              minutesSaved: part.data.minutesSaved,
              stopped: part.data.stopped
            }
          }
        } else if (part.event == 'messageUpdated') {
          yield {
            messageUpdated: {
              messageId: part.data.messageId,
              minutesSaved: part.data.minutesSaved
            }
          }
        } else if (part.event == 'threadUpdated') {
          yield { threadUpdated: { name: part.data.name } }
        } else if (part.event == 'status') {
          yield {
            status: {
              action: part.data.action,
              toolName: part.data.toolName,
              step: part.data.step,
              description: part.data.description,
              args: part.data.args,
              result: part.data.result
            }
          }
        }
      }
    } else {
      yield resp.json()
    }
  }

  private addFileToForm(file: File, formData: FormData, name: string) {
    // on windows it is not properly solving the mime type of markdown files
    if (file.name.toLowerCase().endsWith('.md')) {
      const markdownFile = new File([file], file.name, { type: 'text/markdown' })
      formData.append(name, markdownFile)
    } else {
      formData.append(name, file)
    }
  }

  private async *fetchSSEStream<T = any>(resp: Response, url: string): AsyncIterable<SSEPayload<T>> {
    const reader = resp.body!.getReader()
    let done = false

    while (!done) {
      const { value, done: readerDone } = await reader.read()
      done = readerDone
      if (!value) return

      const events = ServerSentEvent.fromBytes(value)
      for (const ev of events) {
        if (ev.event === 'error') {
          console.warn(`Error event sent by server in response to ${url}`, ev)
          throw new HttpError(resp.status, ev.data)
        }
        if (ev.event === 'heartbeat') {
          continue
        }
        yield {
          event: ev.event || 'message',
          data: ev.event && ev.data ? (JSON.parse(ev.data) as T) : (ev.data as T)
        }
      }
    }
  }

  async transcribeAudio(threadId: number, blob: Blob): Promise<string> {
    const formData = new FormData()
    const file = new File([blob], 'audio.webm', { type: 'audio/webm' })
    formData.append('file', file)

    const data = await this.post(`/threads/${threadId}/transcriptions`, formData)
    return data.transcription
  }

  async completeToolOauthAuth(toolId: string, agentId: number, state: string, code: string): Promise<void> {
    await this.put(`/tools/${toolId}/agents/${agentId}/auth`, { state: state, code: code })
  }

  async deleteToolAuth(toolId: string, state: string): Promise<void> {
    await this.delete(`/tools/${toolId}/oauth/${state}`)
  }

  async completeToolAuthTokenAuth(toolId: string, agentId: number, authToken: string): Promise<void> {
    await this.put(`/tools/${toolId}/agents/${agentId}/auth`, { auth_token: authToken })
  }

  async findBudgetUsage(): Promise<BudgetUsage> {
    return await this.fetchJson('/budget')
  }

  async findUserProfile(): Promise<UserProfile> {
    return await this.fetchJson(`/users/current`)
  }

  async findUsers(): Promise<User[]> {
    return await this.fetchJson('/users')
  }

  async findTeamUsers(teamId: number, limit: number = 20, offset: number = 0, search: string = ''): Promise<TeamUser[]> {
    const params = new URLSearchParams({ limit: limit.toString(), offset: offset.toString() })
    if (search) params.set('search', search)
    return await this.fetchJson(`/teams/${teamId}/users?${params.toString()}`)
  }

  async findTeams(): Promise<Team[]> {
    return await this.fetchJson('/teams')
  }

  async createTeam(team: TeamCreate): Promise<Team> {
    return await this.post('/teams', team)
  }

  async updateTeam(teamId: number, updated: TeamUpdate): Promise<Team> {
    return await this.put(`/teams/${teamId}`, updated)
  }

  async deleteTeam(teamId: number): Promise<void> {
    await this.delete(`/teams/${teamId}`)
  }

  async addUsersToTeam(teamId: number, users: NewUser[]): Promise<void> {
    await this.post(`/teams/${teamId}/users`, users)
  }

  async updateUserRoleInTeam(teamId: number, userId: number, role: Role): Promise<void> {
    await this.put(`/teams/${teamId}/users/${userId}`, { role })
  }

  async deleteUserFromTeam(teamId: number, userId: number): Promise<void> {
    await this.delete(`/teams/${teamId}/users/${userId}`)
  }

  async acceptTeamInvitation(teamId: number): Promise<void> {
    await this.put(`/users/current/teams/${teamId}`)
  }

  async rejectTeamInvitation(teamId: number): Promise<void> {
    await this.delete(`/users/current/teams/${teamId}`)
  }

  async findExternalAgents(): Promise<ExternalAgent[]> {
    return await this.fetchJson('/external-agents')
  }

  async addExternalAgent(externalAgent: NewExternalAgent): Promise<ExternalAgent> {
    return await this.post('/external-agents', externalAgent)
  }

  async addExternalAgentTimeSaving(externalAgentId: number, date: Date, minutesSaved: number): Promise<ExternalAgentTimeSaving> {
    return await this.post(`/external-agents/${externalAgentId}/time-savings`, { date: moment(date).utc(), minutesSaved })
  }
}

export class ThreadMessagePart {
  userMessage?: { id: number, files: UploadedFile[] }
  answerText?: string
  metadata?: {
    answerMessageId?: number
    files: UploadedFile[],
    minutesSaved?: number
    stopped?: boolean
  }
  messageUpdated?: {
    messageId: number
    minutesSaved: number
  }
  threadUpdated?: {
    name: string
  }
  status?: {
    action: string
    toolName?: string
    step?: string
    description?: string
    args?: string
    result?: string | string[]
  }
}

type SSEPayload<T> = {
  /** the event name (empty string for the default "message" event) */
  event: string;
  /** parsed data payload */
  data: T;
};

class ServerSentEvent {
  event?: string
  data: string

  constructor(data: string, event?: string) {
    this.data = data
    this.event = event
  }

  public static fromBytes(bs: Uint8Array): ServerSentEvent[] {
    const text = new TextDecoder('utf-8').decode(bs)
    const events = text.split(/\r\n\r\n/)
    return events.map((e) => ServerSentEvent.parseEvent(e))
  }

  private static parseEvent(event: string): ServerSentEvent {
    const parts = event.split(/\r\n/)
    let eventType
    const eventPrefix = 'event: '
    if (parts[0].startsWith(eventPrefix)) {
      eventType = parts[0].substring(eventPrefix.length)
    }
    const dataPrefix = 'data: '
    const data = parts
      .filter((p) => p.startsWith(dataPrefix))
      .map((p) => p.substring(dataPrefix.length))
      .join('\n')
    return new ServerSentEvent(data, eventType)
  }
}
