import { useAppStore } from '../stores/appStore'

interface Props {
  cleanupStarting?: boolean
}

/**
 * Shared status indicator for Smart Cleanup. Reflects whichever cleanup
 * provider is currently active:
 *  - local: "Active" / "Not running" / "Model needed" depending on llm state
 *  - openai_compatible / anthropic / gemini: "Cloud key set" / "API key needed"
 */
export function CleanupStatusBadge({ cleanupStarting = false }: Props) {
  const aiCleanup = useAppStore((s) => s.aiCleanup)
  const cleanupProvider = useAppStore((s) => s.cleanupProvider)
  const cleanupHasKey = useAppStore((s) => s.cleanupHasKey)
  const llmReady = useAppStore((s) => s.llmReady)
  const llmDownloaded = useAppStore((s) => s.llmDownloaded)

  if (!aiCleanup) return null

  const usingCloud = cleanupProvider !== 'local'
  const hasKey = usingCloud
    ? cleanupHasKey[cleanupProvider as 'openai_compatible' | 'anthropic' | 'gemini']
    : false

  let dotClass: string
  let textClass: string
  let label: string

  if (cleanupStarting) {
    dotClass = 'bg-chirp-amber-400 animate-pulse'
    textClass = 'text-dm-secondary'
    label = 'Getting ready...'
  } else if (usingCloud) {
    if (hasKey) {
      dotClass = 'bg-chirp-success'
      textClass = 'text-dm-secondary'
      label = 'Cloud key set'
    } else {
      dotClass = 'bg-chirp-amber-400'
      textClass = 'text-chirp-amber-500'
      label = 'API key needed'
    }
  } else if (llmReady) {
    dotClass = 'bg-chirp-success'
    textClass = 'text-dm-secondary'
    label = 'Active'
  } else if (llmDownloaded) {
    dotClass = 'bg-chirp-amber-400'
    textClass = 'text-chirp-amber-500'
    label = 'Not running'
  } else {
    dotClass = 'bg-chirp-amber-400'
    textClass = 'text-chirp-amber-500'
    label = 'Model needed'
  }

  return (
    <span className="flex items-center gap-1">
      <div className={`h-1.5 w-1.5 rounded-full ${dotClass}`} />
      <span className={`text-[11px] ${textClass}`}>{label}</span>
    </span>
  )
}

