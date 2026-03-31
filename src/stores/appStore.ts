import { create } from 'zustand'
import { DEFAULT_SETTINGS, type ErrorType } from '../lib/constants'

export type AppStatus = 'idle' | 'listening' | 'processing' | 'polishing' | 'done' | 'error'
export type SttModel = 'parakeet-tdt-0.6b'

export interface DictionaryEntry {
  from: string
  to: string
}

export interface SnippetEntry {
  trigger: string
  expansion: string
}

export interface VocabularyEntry {
  word: string
  boost: number
}

export interface TranscriptionEntry {
  text: string
  timestamp: string
  wordCount: number
  durationMs: number
  speechDurationMs: number
  wasCleanedUp?: boolean
}

export interface AppState {
  // Recording state
  status: AppStatus
  errorType: ErrorType | null
  wordCount: number | null
  amplitudes: number[]

  // Settings
  hotkey: string
  launchAtLogin: boolean
  playSoundOnComplete: boolean
  autoDismissOverlay: boolean
  smartFormatting: boolean

  // Audio
  inputDevice: string
  inputLevel: number

  // Model
  model: SttModel
  modelDownloaded: Record<string, boolean>
  modelDownloadProgress: number | null

  // AI Cleanup
  aiCleanup: boolean
  llmReady: boolean

  // Beam Search
  beamSearch: boolean
  llmDownloadProgress: number | null

  // Dictionary
  dictionary: DictionaryEntry[]

  // Snippets
  snippets: SnippetEntry[]

  // Vocabulary
  vocabulary: VocabularyEntry[]

  // History
  history: TranscriptionEntry[]

  // Onboarding
  onboardingComplete: boolean

  // Tone
  toneMode: string

  // Overlay
  overlayPosition: 'bottom' | 'top'
  showPassiveOverlay: boolean

  // History retention
  historyRetentionDays: number

  // Telemetry
  helpImprove: boolean

  // Hotkey status
  hotkeyStatus: 'idle' | 'retrying' | 'active' | 'failed' | 'accessibility_required'

  // Settings saved indicator
  settingsSaved: boolean

  // About modal
  aboutModalOpen: boolean

  // Loading
  settingsLoaded: boolean

  // Settings page
  settingsPage: string

  // Actions
  setStatus: (status: AppStatus) => void
  setError: (errorType: ErrorType) => void
  setAmplitudes: (data: number[]) => void
  setWordCount: (count: number) => void
  setInputLevel: (level: number) => void
  setModelDownloadProgress: (progress: number | null) => void
  setLlmDownloadProgress: (progress: number | null) => void
  setLlmReady: (ready: boolean) => void
  updateSettings: (partial: Partial<AppState>) => void
  addDictionaryEntry: (from: string, to: string) => void
  removeDictionaryEntry: (index: number) => void
  setSnippets: (snippets: SnippetEntry[]) => void
  addSnippet: (trigger: string, expansion: string) => void
  updateSnippet: (index: number, trigger: string, expansion: string) => void
  removeSnippet: (index: number) => void
  addVocabularyEntry: (word: string, boost?: number) => void
  removeVocabularyEntry: (index: number) => void
  updateVocabularyBoost: (index: number, boost: number) => void
  setVocabulary: (vocabulary: VocabularyEntry[]) => void
  setSettingsLoaded: () => void
  setHistory: (history: TranscriptionEntry[]) => void
  removeHistoryEntry: (timestamp: string) => void
  setSettingsPage: (page: string) => void
  setHotkeyStatus: (status: 'idle' | 'retrying' | 'active' | 'failed' | 'accessibility_required') => void
  setOnboardingComplete: (complete: boolean) => void
  setSettingsSaved: (saved: boolean) => void
  setAboutModalOpen: (open: boolean) => void
}

export const useAppStore = create<AppState>((set) => ({
  // Recording state
  status: 'idle',
  errorType: null,
  wordCount: null,
  amplitudes: [],

  // Settings (from defaults)
  hotkey: DEFAULT_SETTINGS.hotkey,
  launchAtLogin: DEFAULT_SETTINGS.launchAtLogin,
  playSoundOnComplete: DEFAULT_SETTINGS.playSoundOnComplete,
  autoDismissOverlay: DEFAULT_SETTINGS.autoDismissOverlay,
  smartFormatting: DEFAULT_SETTINGS.smartFormatting,

  // Audio
  inputDevice: DEFAULT_SETTINGS.inputDevice,
  inputLevel: 0,

  // Model
  model: DEFAULT_SETTINGS.model,
  modelDownloaded: {},
  modelDownloadProgress: null,

  // AI Cleanup
  aiCleanup: DEFAULT_SETTINGS.aiCleanup,
  llmReady: false,
  llmDownloadProgress: null,

  // Beam Search
  beamSearch: DEFAULT_SETTINGS.beamSearch,

  // Dictionary
  dictionary: [],

  // Snippets
  snippets: [],

  // Vocabulary
  vocabulary: [],

  // History
  history: [],

  // Onboarding
  onboardingComplete: DEFAULT_SETTINGS.onboardingComplete,

  // Tone
  toneMode: DEFAULT_SETTINGS.toneMode,

  // Overlay
  overlayPosition: DEFAULT_SETTINGS.overlayPosition,
  showPassiveOverlay: DEFAULT_SETTINGS.showPassiveOverlay,

  // History retention
  historyRetentionDays: DEFAULT_SETTINGS.historyRetentionDays,

  // Telemetry
  helpImprove: DEFAULT_SETTINGS.helpImprove,

  // Hotkey status
  hotkeyStatus: 'idle',

  // Settings saved indicator
  settingsSaved: false,

  // About modal
  aboutModalOpen: false,

  // Loading
  settingsLoaded: false,

  // Settings page
  settingsPage: 'home',

  // Actions
  setStatus: (status) => set({ status, errorType: status !== 'error' ? null : undefined }),
  setError: (errorType) => set({ status: 'error', errorType }),
  setAmplitudes: (amplitudes) => set({ amplitudes }),
  setWordCount: (wordCount) => set({ wordCount }),
  setInputLevel: (inputLevel) => set({ inputLevel }),
  setModelDownloadProgress: (modelDownloadProgress) => set({ modelDownloadProgress }),
  setLlmDownloadProgress: (llmDownloadProgress) => set({ llmDownloadProgress }),
  setLlmReady: (llmReady) => set({ llmReady }),
  updateSettings: (partial) => set(partial),
  addDictionaryEntry: (from, to) =>
    set((state) => ({ dictionary: [...state.dictionary, { from, to }] })),
  removeDictionaryEntry: (index) =>
    set((state) => ({ dictionary: state.dictionary.filter((_, i) => i !== index) })),
  setSnippets: (snippets) => set({ snippets }),
  addSnippet: (trigger, expansion) =>
    set((state) => ({ snippets: [...state.snippets, { trigger, expansion }] })),
  updateSnippet: (index, trigger, expansion) =>
    set((state) => ({
      snippets: state.snippets.map((s, i) => (i === index ? { trigger, expansion } : s)),
    })),
  removeSnippet: (index) =>
    set((state) => ({ snippets: state.snippets.filter((_, i) => i !== index) })),
  addVocabularyEntry: (word, boost = 3.0) =>
    set((state) => ({ vocabulary: [...state.vocabulary, { word, boost }] })),
  removeVocabularyEntry: (index) =>
    set((state) => ({ vocabulary: state.vocabulary.filter((_, i) => i !== index) })),
  updateVocabularyBoost: (index, boost) =>
    set((state) => ({
      vocabulary: state.vocabulary.map((v, i) => (i === index ? { ...v, boost } : v)),
    })),
  setVocabulary: (vocabulary) => set({ vocabulary }),
  setSettingsLoaded: () => set({ settingsLoaded: true }),
  setHistory: (history) => set({ history }),
  removeHistoryEntry: (timestamp) =>
    set((state) => ({ history: state.history.filter((e) => e.timestamp !== timestamp) })),
  setSettingsPage: (settingsPage) => set({ settingsPage }),
  setHotkeyStatus: (hotkeyStatus) => set({ hotkeyStatus }),
  setOnboardingComplete: (onboardingComplete) => set({ onboardingComplete }),
  setSettingsSaved: (settingsSaved) => set({ settingsSaved }),
  setAboutModalOpen: (aboutModalOpen) => set({ aboutModalOpen }),
}))
