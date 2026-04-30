import { useCallback, useEffect, useMemo, useState } from 'react'
import { invoke } from '@tauri-apps/api/core'
import { Copy, Download, Search, Trash2 } from 'lucide-react'
import { useAppStore } from '../../stores/appStore'
import { useTauri } from '../../hooks/useTauri'
import type { TranscriptionEntry } from '../../stores/appStore'
import type { DictationPatterns } from '../home/types'

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

function formatTime(timestamp: string): string {
  return new Date(timestamp)
    .toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' })
    .replace(' AM', 'a')
    .replace(' PM', 'p')
    .toLowerCase()
}

function formatDate(timestamp: string): string {
  const d = new Date(timestamp)
  const today = new Date()
  const yesterday = new Date()
  yesterday.setDate(today.getDate() - 1)
  if (d.toDateString() === today.toDateString()) return 'Today'
  if (d.toDateString() === yesterday.toDateString()) return 'Yesterday'
  return d.toLocaleDateString([], { weekday: 'long', month: 'long', day: 'numeric' })
}

function formatDuration(ms: number): string {
  const seconds = Math.round(ms / 1000)
  if (seconds < 60) return `${seconds}s`
  const m = Math.floor(seconds / 60)
  const s = seconds % 60
  if (m < 60) return `${m}m ${s}s`
  const h = Math.floor(m / 60)
  return `${h}h ${m % 60}m`
}

interface DaySection {
  key: string
  label: string
  entries: TranscriptionEntry[]
  totalWords: number
}

export function HistoryPage() {
  const tauri = useTauri()
  const history = useAppStore((s) => s.history)
  const setHistory = useAppStore((s) => s.setHistory)
  const [search, setSearch] = useState('')
  const [appNameMap, setAppNameMap] = useState<Record<string, string>>({})

  useEffect(() => {
    invoke<DictationPatterns>('get_dictation_patterns', { period: 'all' })
      .then((p) => {
        const map: Record<string, string> = {}
        for (const a of p.topApps) map[a.raw.toLowerCase()] = a.display
        setAppNameMap(map)
      })
      .catch(() => setAppNameMap({}))
  }, [history.length])

  const resolveAppDisplay = useCallback(
    (raw: string) => appNameMap[raw.toLowerCase()] ?? fallbackDisplayName(raw),
    [appNameMap],
  )

  const filtered = useMemo(() => {
    const q = search.trim().toLowerCase()
    if (!q) return history
    return history.filter((e) => e.text.toLowerCase().includes(q))
  }, [history, search])

  const sections = useMemo<DaySection[]>(() => {
    const sorted = [...filtered].sort(
      (a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime(),
    )
    const groups = new Map<string, DaySection>()
    for (const e of sorted) {
      const d = new Date(e.timestamp)
      const key = `${d.getFullYear()}-${d.getMonth() + 1}-${d.getDate()}`
      let g = groups.get(key)
      if (!g) {
        g = { key, label: formatDate(e.timestamp), entries: [], totalWords: 0 }
        groups.set(key, g)
      }
      g.entries.push(e)
      g.totalWords += e.wordCount
    }
    return Array.from(groups.values())
  }, [filtered])

  const totals = useMemo(() => {
    return filtered.reduce(
      (acc, e) => ({
        words: acc.words + e.wordCount,
        sessions: acc.sessions + 1,
        durationMs: acc.durationMs + e.durationMs,
      }),
      { words: 0, sessions: 0, durationMs: 0 },
    )
  }, [filtered])

  const handleCopy = useCallback((entry: TranscriptionEntry) => {
    navigator.clipboard.writeText(entry.text).catch(() => { /* clipboard denied */ })
  }, [])

  const handleDelete = useCallback(async (entry: TranscriptionEntry) => {
    try {
      await tauri.deleteHistoryEntry(entry.timestamp)
      const next = await tauri.getHistory()
      setHistory(next)
    } catch {
      /* ignore */
    }
  }, [tauri, setHistory])

  const handleExport = useCallback(() => {
    const blob = new Blob([JSON.stringify(history, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `chirp-history-${new Date().toISOString().slice(0, 10)}.json`
    a.click()
    URL.revokeObjectURL(url)
  }, [history])

  return (
    <div className="mx-auto max-w-[1080px] px-12 pt-10">
      <header className="mb-8 flex items-center gap-3">
        <div className="relative flex-1">
          <Search size={14} className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 text-white/35" />
          <input
            type="search"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search dictations"
            className="w-full rounded-full border border-white/10 bg-white/[0.03] py-2 pl-9 pr-4 font-geist text-[13px] text-white placeholder:text-white/30 focus:border-white/30 focus:outline-none"
          />
        </div>
        <button
          type="button"
          onClick={handleExport}
          className="flex items-center gap-2 rounded-full border border-white/10 bg-white/[0.03] px-4 py-2 font-geist text-[12px] text-white/85 transition-colors hover:bg-white/[0.07]"
        >
          <Download size={13} />
          Export
        </button>
      </header>

      {sections.length === 0 ? (
        <div className="rounded-2xl border border-white/[0.06] bg-white/[0.02] p-12 text-center font-geist text-[13px] text-white/45">
          {search ? 'No dictations match your search.' : 'No dictations yet.'}
        </div>
      ) : (
        <div className="flex flex-col gap-10">
          {sections.map((section) => (
            <section key={section.key} className="card-surface overflow-hidden">
              <div className="flex items-baseline justify-between border-b border-white/[0.06] px-5 py-3">
                <h3 className="font-geist text-[10px] font-medium uppercase tracking-[0.2em] text-white/45">
                  {section.label}
                </h3>
                <span
                  className="font-geist-mono text-[11px] text-white/40"
                  style={{ fontFeatureSettings: '"tnum"' }}
                >
                  {section.entries.length} sessions · {section.totalWords.toLocaleString()} words
                </span>
              </div>
              <ul className="divide-y divide-white/[0.06] px-5">
                {section.entries.map((entry) => {
                  const appDisplay = entry.targetApp ? resolveAppDisplay(entry.targetApp) : null
                  return (
                    <li
                      key={entry.timestamp}
                      className="group grid grid-cols-[64px_1fr_auto] items-start gap-4 py-3"
                    >
                      <span
                        className="pt-0.5 font-geist-mono text-[11px] text-white/35"
                        style={{ fontFeatureSettings: '"tnum"' }}
                      >
                        {formatTime(entry.timestamp)}
                      </span>
                      <div className="flex min-w-0 flex-col gap-1">
                        <p className="break-words font-geist text-[13px] text-white/85">
                          {entry.text}
                        </p>
                        <div className="flex items-center gap-2 font-geist text-[11px] text-white/40">
                          {appDisplay && <span>{appDisplay}</span>}
                          {appDisplay && <span className="text-white/20">·</span>}
                          <span>{formatDuration(entry.durationMs)}</span>
                          <span className="text-white/20">·</span>
                          <span>{entry.wordCount} words</span>
                          {entry.wasCleanedUp && (
                            <>
                              <span className="text-white/20">·</span>
                              <span className="text-chirp-yellow">polished</span>
                            </>
                          )}
                        </div>
                      </div>
                      <div className="flex items-center gap-1 opacity-0 transition-opacity group-hover:opacity-100">
                        <button
                          type="button"
                          onClick={() => handleCopy(entry)}
                          aria-label="Copy text"
                          className="flex h-7 w-7 items-center justify-center rounded-full text-white/45 transition-colors hover:bg-white/[0.06] hover:text-white"
                        >
                          <Copy size={13} />
                        </button>
                        <button
                          type="button"
                          onClick={() => handleDelete(entry)}
                          aria-label="Delete entry"
                          className="flex h-7 w-7 items-center justify-center rounded-full text-white/45 transition-colors hover:bg-red-500/10 hover:text-red-400"
                        >
                          <Trash2 size={13} />
                        </button>
                      </div>
                    </li>
                  )
                })}
              </ul>
            </section>
          ))}

          <footer className="sticky bottom-6 z-10 -mx-12 mt-4 border-t border-white/[0.06] bg-black/70 px-12 py-4 backdrop-blur-md">
            <div className="flex items-center justify-between font-geist text-[11px] uppercase tracking-[0.18em] text-white/45">
              <span>Total</span>
              <span
                className="font-geist-mono normal-case tracking-normal text-white/85"
                style={{ fontFeatureSettings: '"tnum"' }}
              >
                {totals.words.toLocaleString()} words · {totals.sessions} sessions ·{' '}
                {formatDuration(totals.durationMs)}
              </span>
            </div>
          </footer>
        </div>
      )}
    </div>
  )
}
