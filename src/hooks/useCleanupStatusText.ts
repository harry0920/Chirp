import { useAppStore } from '../stores/appStore'

/**
 * Resolves a single status string for the Smart Cleanup feature, regardless of
 * which provider is active. Used in the tray popup where we render the status
 * inline with other text rather than as a dot+label badge.
 */
export function useCleanupStatusText(cleanupStarting = false): string {
  const aiCleanup = useAppStore((s) => s.aiCleanup)
  const cleanupProvider = useAppStore((s) => s.cleanupProvider)
  const cleanupHasKey = useAppStore((s) => s.cleanupHasKey)
  const llmReady = useAppStore((s) => s.llmReady)
  const llmDownloaded = useAppStore((s) => s.llmDownloaded)

  if (!aiCleanup) return 'Off'
  if (cleanupStarting) return 'Starting...'

  if (cleanupProvider !== 'local') {
    return cleanupHasKey[cleanupProvider as 'openai_compatible' | 'anthropic' | 'gemini']
      ? 'Cloud key set'
      : 'API key needed'
  }
  if (llmReady) return 'Active'
  if (llmDownloaded) return 'Not running'
  return 'Model needed'
}
