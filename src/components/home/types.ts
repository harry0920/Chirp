export type Period = 'week' | 'month' | 'year' | 'all'

export interface DailyPoint {
  date: string
  words: number
  sessions: number
  durationMs: number
}

export interface AppPoint {
  raw: string
  display: string
  count: number
  words: number
  percent: number
}

export interface DictationPatterns {
  period: Period
  hourlyGrid: number[][]
  daily: DailyPoint[]
  topApps: AppPoint[]
  totalWords: number
  totalSessions: number
  totalDurationMs: number
}

export type AttentionSeverity = 'info' | 'warning' | 'error'

export interface AttentionItem {
  id: string
  severity: AttentionSeverity
  message: string
  action: string | null
}
