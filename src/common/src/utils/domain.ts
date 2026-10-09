export enum FileStatus {
    PENDING = 'PENDING',
    PROCESSED = 'PROCESSED',
    ERROR = 'ERROR',
    QUOTA_EXCEEDED = 'QUOTA_EXCEEDED',
    RATE_LIMITED = 'RATE_LIMITED'
  }
  
export class UploadedFile {
    name: string
    contentType: string
    status?: FileStatus
    file?: File
    id?: number

    constructor(name: string, contentType: string, status?: FileStatus, file?: File, id?: number) {
        this.name = name
        this.contentType = contentType
        this.status = status
        this.file = file
        this.id = id
    }
}

export class AgentPrompt {
    id?: number
    name?: string
    content?: string
    shared: boolean
    canEdit: boolean
    starter: boolean
    collapsed: boolean
  
    constructor(id?: number, name?: string, content?: string, shared: boolean = false, canEdit: boolean = true, starter: boolean = false, collapsed: boolean = false) {
      this.id = id
      this.name = name
      this.content = content
      this.shared = shared
      this.canEdit = canEdit
      this.starter = starter
      this.collapsed = collapsed
    }
}

export class UserFeedback {
    isPositive: boolean
    minutesSaved: number
    text?: string
  
    constructor(isPositive: boolean, minutesSaved: number, text?: string) {
      this.isPositive = isPositive;
      this.minutesSaved = minutesSaved;
      this.text = text;
    }
}
  