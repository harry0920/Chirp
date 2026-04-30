import { useCallback, useEffect, useMemo, useState } from 'react'
import { invoke } from '@tauri-apps/api/core'
import { useAppStore } from '../../stores/appStore'
import { useTauri } from '../../hooks/useTauri'
import { HeroMetric } from '../home/HeroMetric'
import { PatternsRow } from '../home/PatternsRow'
import { RecentsRow } from '../home/RecentsRow'
import { ReadinessPill } from '../home/ReadinessPill'
import { AttentionStrip } from '../home/AttentionStrip'
import { TestDictationCard } from '../home/TestDictationCard'
import type {
  AttentionItem,
  DictationPatterns,
  Period,
} from '../home/types'

const APP_NAMES_FALLBACK: Record<string, string> = {
  'slack.exe': 'Slack',
  'discord.exe': 'Discord',
  'code.exe': 'VS Code',
  'cursor.exe': 'Cursor',
  'chrome.exe': 'Chrome',
  'msedge.exe': 'Edge',
  'firefox.exe': 'Firefox',
  'notion.exe': 'Notion',
  'linear.exe': 'Linear',
  'obsidian.exe': 'Obsidian',
}

function fallbackDisplayName(raw: string): string {
  const key = raw.trim().toLowerCase()
  if (APP_NAMES_FALLBACK[key]) return APP_NAMES_FALLBACK[key]
  const stripped = key.replace(/\.(exe|app)$/i, '')
  return stripped
    .split(/[\s_\-.]+/)
    .filter(Boolean)
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
    .join(' ')
}

export function HomePage() {
  const tauri = useTauri()
  const history = useAppStore((s) => s.history)
  const setHistory = useAppStore((s) => s.setHistory)
  const setSettingsPage = useAppStore((s) => s.setSettingsPage)

  const [period, setPeriod] = useState<Period>('month')
  const [patterns, setPatterns] = useState<DictationPatterns | null>(null)
  const [attention, setAttention] = useState<AttentionItem[]>([])

  // Resolve display names via the backend's mapping. Cached on the
  // top apps array so we don't round-trip per row.
  const appNameMap = useMemo(() => {
    const map: Record<string, string> = {}
    if (patterns) {
      for (const app of patterns.topApps) {
        map[app.raw.toLowerCase()] = app.display
      }
    }
    return map
  }, [patterns])

  const resolveAppDisplay = useCallback(
    (raw: string) => {
      const key = raw.toLowerCase()
      return appNameMap[key] ?? fallbackDisplayName(raw)
    },
    [appNameMap],
  )

  useEffect(() => {
    let cancelled = false
    invoke<DictationPatterns>('get_dictation_patterns', { period })
      .then((p) => { if (!cancelled) setPatterns(p) })
      .catch(() => { if (!cancelled) setPatterns(null) })
    invoke<AttentionItem[]>('get_attention_items')
      .then((items) => { if (!cancelled) setAttention(items) })
      .catch(() => { if (!cancelled) setAttention([]) })
    return () => { cancelled = true }
  }, [period, history.length])

  const recents = useMemo(() => {
    return [...history]
      .sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime())
      .slice(0, 3)
  }, [history])

  const longestDictationWords = useMemo(() => {
    return history.reduce((max, e) => Math.max(max, e.wordCount), 0)
  }, [history])

  const handleCopy = useCallback(async (entry: { text: string }) => {
    try {
      await navigator.clipboard.writeText(entry.text)
    } catch {
      /* clipboard denied — silently ignore */
    }
  }, [])

  const handleDelete = useCallback(async (entry: { timestamp: string }) => {
    try {
      await tauri.deleteHistoryEntry(entry.timestamp)
      const next = await tauri.getHistory()
      setHistory(next)
    } catch {
      /* ignore — backend will surface its own error */
    }
  }, [tauri, setHistory])

  const handleAttentionAction = useCallback((item: AttentionItem) => {
    if (!item.action) return
    if (item.action.startsWith('page:')) {
      const id = item.action.slice('page:'.length)
      setSettingsPage(id)
    } else if (item.action.startsWith('settings:')) {
      setSettingsPage('settings')
    }
  }, [setSettingsPage])

  const handleViewAll = useCallback(() => {
    // History page lands in Phase 6 — for now this is a no-op slot.
    setSettingsPage('history')
  }, [setSettingsPage])

  const isEmpty = history.length === 0

  return (
    <div className="relative">
      <div className="absolute right-0 top-0 z-10">
        <ReadinessPill />
      </div>

      <div className="flex flex-col gap-10">
        {attention.length > 0 && (
          <AttentionStrip items={attention} onAction={handleAttentionAction} />
        )}

        <HeroMetric
          daily={patterns?.daily ?? []}
          totalWords={patterns?.totalWords ?? 0}
          period={period}
          onPeriodChange={setPeriod}
        />

        <PatternsRow
          hourlyGrid={patterns?.hourlyGrid ?? Array.from({ length: 7 }, () => Array(24).fill(0))}
          topApps={patterns?.topApps ?? []}
          totalWords={patterns?.totalWords ?? 0}
          totalSessions={patterns?.totalSessions ?? 0}
          longestDictationWords={longestDictationWords}
          totalDurationMs={patterns?.totalDurationMs ?? 0}
        />

        {isEmpty ? (
          <TestDictationCard />
        ) : (
          <RecentsRow
            entries={recents}
            onCopy={handleCopy}
            onDelete={handleDelete}
            onViewAll={handleViewAll}
            resolveAppDisplay={resolveAppDisplay}
          />
        )}
      </div>
    </div>
  )
}
