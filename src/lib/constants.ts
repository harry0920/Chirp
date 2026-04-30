export const DEFAULT_SETTINGS = {
  hotkey: 'MetaLeft+ShiftLeft+Space',
  hotkeyMode: 'hold' as const,
  launchAtLogin: true,
  playSoundOnComplete: false,
  autoDismissOverlay: true,
  smartFormatting: true,
  inputDevice: 'default',
  model: 'parakeet-tdt-0.6b' as const,
  onboardingComplete: false,
  aiCleanup: true,
  overlayPosition: 'default' as string | { x: number; y: number },
  toneMode: 'message',
  historyRetentionDays: 0,
  helpImprove: false,
  beamSearch: false,
  cleanupModel: 'chirp-v2' as string,
  cleanupProvider: 'local' as 'local' | 'openai_compatible' | 'anthropic' | 'gemini',
  cleanupProviderConfigs: {
    openai_compatible: { model: 'gpt-4.1-mini', baseUrl: 'https://api.openai.com/v1' },
    anthropic: { model: 'claude-haiku-4-5' },
    gemini: { model: 'gemini-2.5-flash' },
  } as Record<'openai_compatible' | 'anthropic' | 'gemini', { model: string; baseUrl?: string }>,
  darkMode: false,
}

export const CLEANUP_PROVIDERS = [
  { id: 'local', label: 'Local', description: 'Runs entirely on this device' },
  { id: 'openai_compatible', label: 'OpenAI-compatible', description: 'OpenAI, Groq, OpenRouter, Together, Ollama, etc.' },
  { id: 'anthropic', label: 'Anthropic (Claude)', description: 'Cloud cleanup via Anthropic\'s API' },
  { id: 'gemini', label: 'Google Gemini', description: 'Cloud cleanup via Google AI Studio' },
] as const

export type CleanupProviderId = typeof CLEANUP_PROVIDERS[number]['id']

export const TONE_MODES = [
  { id: 'message', label: 'Message', description: 'Natural conversational tone' },
  { id: 'email', label: 'Email', description: 'Professional email formatting' },
] as const

export const STT_MODELS = [
  { id: 'parakeet-tdt-0.6b' as const, name: 'Parakeet TDT — NVIDIA', size: '465 MB', description: 'Best accuracy · 25 languages · fast on any PC', recommended: true },
]

export const LLM_MODEL = {
  name: 'Qwen 3 1.7B',
  displayName: 'Smart Cleanup',
  size: '1.1 GB',
  friendlySize: 'About 1.1 GB',
  description: 'On-device language model that removes filler words, fixes stutters, and resolves self-corrections — all without sending audio anywhere.',
  attribution: 'Built on Qwen 3 1.7B — Alibaba (Apache 2.0)',
}

export const CLEANUP_EXAMPLE = {
  before: "so um basically I was thinking that we should like probably move the meeting to uh Thursday if that works",
  after: "I was thinking we should probably move the meeting to Thursday, if that works.",
}

export const ERROR_MESSAGES = {
  mic_not_found: {
    title: 'No microphone detected',
    help: 'Connect a microphone and try again',
    action: null,
  },
  mic_permission: {
    title: "Couldn't access microphone",
    help: 'Check your system permissions',
    action: { label: 'Open Settings', type: 'os_settings' as const },
  },
  model_not_loaded: {
    title: 'Speech model not ready',
    help: 'Download a model in settings',
    action: { label: 'Open Settings', type: 'app_settings' as const },
  },
  transcription_failed: {
    title: "Couldn't process audio",
    help: 'Try speaking more clearly',
    action: { label: 'Try Again', type: 'retry' as const },
  },
  injection_failed: {
    title: "Couldn't paste text",
    help: 'Make sure a text field is focused',
    action: { label: 'Copy to Clipboard', type: 'copy' as const },
  },
  accessibility_denied: {
    title: 'Accessibility access needed',
    help: 'Enable Chirp in System Settings > Privacy > Accessibility',
    action: { label: 'Open Settings', type: 'os_settings' as const },
  },
  unknown: {
    title: 'Something went wrong',
    help: 'Please try again',
    action: { label: 'Try Again', type: 'retry' as const },
  },
} as const

export type ErrorType = keyof typeof ERROR_MESSAGES
