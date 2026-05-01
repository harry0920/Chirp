import { useCallback, useMemo, useState } from 'react'
import { Copy, Download, Search, Trash2, X } from 'lucide-react'
import type { TranscriptionEntry } from '../../stores/appStore'

interface Props {
  entries: TranscriptionEntry[]
  onCopy: (entry: TranscriptionEntry) => void
  onDelete: (entry: TranscriptionEntry) => void
  resolveAppDisplay: (raw: string) => string
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

/** Sub-second precision for short durations — "4.2s" reads better
 *  than rounding to "4s" when comparing speech vs processing in the
 *  expanded breakdown. Falls back to formatDuration for >= 60s. */
function formatPreciseDuration(ms: number): string {
  if (ms <= 0) return '0s'
  if (ms < 60_000) {
    const s = ms / 1000
    return s < 10 ? `${s.toFixed(1)}s` : `${Math.round(s)}s`
  }
  return formatDuration(ms)
}

function formatFullTimestamp(timestamp: string): string {
  const d = new Date(timestamp)
  const date = d.toLocaleDateString([], {
    month: 'long',
    day: 'numeric',
    year: 'numeric',
  })
  const time = d.toLocaleTimeString([], {
    hour: 'numeric',
    minute: '2-digit',
  })
  return `${date} · ${time}`
}

function computeWpm(wordCount: number, speechDurationMs: number): number | null {
  if (speechDurationMs <= 0 || wordCount <= 0) return null
  const minutes = speechDurationMs / 60_000
  return Math.round(wordCount / minutes)
}

interface DaySection {
  key: string
  label: string
  entries: TranscriptionEntry[]
  totalWords: number
}

export function HomeHistoryList({ entries, onCopy, onDelete, resolveAppDisplay }: Props) {
  const [search, setSearch] = useState('')
  const [expandedTimestamp, setExpandedTimestamp] = useState<string | null>(null)
  const [copiedTimestamp, setCopiedTimestamp] = useState<string | null>(null)

  const filtered = useMemo(() => {
    const q = search.trim().toLowerCase()
    if (!q) return entries
    return entries.filter((e) => e.text.toLowerCase().includes(q))
  }, [entries, search])

  const sections = useMemo<DaySection[]>(() => {
    const seen = new Set<string>()
    const sorted = [...filtered]
      .sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime())
      .filter((e) => {
        if (seen.has(e.timestamp)) return false
        seen.add(e.timestamp)
        return true
      })
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

  const handleExport = useCallback(() => {
    const blob = new Blob([JSON.stringify(entries, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `chirp-history-${new Date().toISOString().slice(0, 10)}.json`
    a.click()
    URL.revokeObjectURL(url)
  }, [entries])

  const handleRowClick = useCallback((timestamp: string) => {
    setExpandedTimestamp((prev) => (prev === timestamp ? null : timestamp))
  }, [])

  const handleCopyClick = useCallback(
    (entry: TranscriptionEntry) => {
      onCopy(entry)
      setCopiedTimestamp(entry.timestamp)
      setTimeout(() => {
        setCopiedTimestamp((current) => (current === entry.timestamp ? null : current))
      }, 1200)
    },
    [onCopy],
  )

  return (
    <section>
      <header className="mb-4 flex items-center gap-3">
        <span className="font-geist text-[10px] font-medium uppercase tracking-[0.2em] text-white/45">
          History
        </span>
        <div className="relative ml-2 flex-1">
          <Search size={14} className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 text-white/35" />
          <input
            type="search"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search dictations"
            className="w-full rounded-full border border-white/10 bg-white/[0.03] py-2 pl-9 pr-9 font-geist text-[12px] text-white placeholder:text-white/30 transition-all duration-150 focus:border-chirp-yellow/50 focus:outline-none focus:ring-1 focus:ring-chirp-yellow/30"
          />
          {search && (
            <button
              type="button"
              onClick={() => setSearch('')}
              aria-label="Clear search"
              className="absolute right-3 top-1/2 flex h-5 w-5 -translate-y-1/2 items-center justify-center rounded-full text-white/45 transition-all duration-150 hover:bg-white/[0.06] hover:text-white active:scale-90"
            >
              <X size={12} />
            </button>
          )}
        </div>
        <button
          type="button"
          onClick={handleExport}
          className="flex items-center gap-2 rounded-full border border-white/10 bg-white/[0.03] px-4 py-2 font-geist text-[12px] text-white/85 transition-all duration-150 hover:bg-white/[0.07] active:scale-95"
        >
          <Download size={13} />
          Export
        </button>
      </header>

      {sections.length === 0 ? (
        <div className="card-surface animate-fade-in flex flex-col items-center gap-2 p-12 text-center">
          <p className="font-geist text-[13px] text-white/45">
            {search ? 'No dictations match your search.' : 'No dictations yet.'}
          </p>
        </div>
      ) : (
        <div className="flex flex-col gap-6">
          {sections.map((section, sectionIdx) => (
            <section
              key={section.key}
              className="card-surface animate-slide-up overflow-hidden"
              style={{ animationDelay: `${sectionIdx * 80}ms` }}
            >
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
                  const expanded = expandedTimestamp === entry.timestamp
                  const copied = copiedTimestamp === entry.timestamp
                  return (
                    <li
                      key={entry.timestamp}
                      className="-mx-5 px-5 transition-colors duration-150 hover:bg-white/[0.02]"
                    >
                      <button
                        type="button"
                        onClick={() => handleRowClick(entry.timestamp)}
                        aria-expanded={expanded}
                        className="grid w-full grid-cols-[64px_1fr] items-start gap-4 py-3 text-left transition-transform duration-100 active:scale-[0.995]"
                      >
                        <span
                          className="pt-0.5 font-geist-mono text-[11px] text-white/35"
                          style={{ fontFeatureSettings: '"tnum"' }}
                        >
                          {formatTime(entry.timestamp)}
                        </span>
                        <div className="flex min-w-0 flex-col gap-1">
                          <p
                            className={`font-geist text-[13px] text-white/85 transition-[max-height] duration-300 ease-out ${
                              expanded ? 'whitespace-pre-wrap break-words' : 'truncate'
                            }`}
                          >
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
                      </button>

                      {/* Expand drawer — grid-rows trick animates between
                          0fr (collapsed) and 1fr (expanded) so the natural
                          content height interpolates smoothly with no JS. */}
                      <div
                        className={`grid grid-cols-[64px_1fr] gap-4 transition-[grid-template-rows,opacity,padding] duration-300 ease-out ${
                          expanded
                            ? 'grid-rows-[1fr] pb-4 opacity-100'
                            : 'grid-rows-[0fr] pb-0 opacity-0'
                        }`}
                      >
                        <span aria-hidden />
                        <div className="overflow-hidden">
                          <ExpandedDetails
                            entry={entry}
                            appDisplay={appDisplay}
                          />
                          <div className="flex items-center gap-2 pt-1">
                            <button
                              type="button"
                              onClick={(e) => {
                                e.stopPropagation()
                                handleCopyClick(entry)
                              }}
                              className="flex items-center gap-1.5 rounded-full border border-white/10 bg-white/[0.03] px-3 py-1.5 font-geist text-[11px] text-white/85 transition-all duration-150 hover:bg-white/[0.07] active:scale-95"
                            >
                              <Copy size={12} />
                              {copied ? 'Copied' : 'Copy'}
                            </button>
                            <button
                              type="button"
                              onClick={(e) => {
                                e.stopPropagation()
                                onDelete(entry)
                              }}
                              className="flex items-center gap-1.5 rounded-full border border-white/10 bg-white/[0.03] px-3 py-1.5 font-geist text-[11px] text-white/85 transition-all duration-150 hover:border-red-500/40 hover:bg-red-500/10 hover:text-red-300 active:scale-95"
                            >
                              <Trash2 size={12} />
                              Delete
                            </button>
                          </div>
                        </div>
                      </div>
                    </li>
                  )
                })}
              </ul>
            </section>
          ))}
        </div>
      )}
    </section>
  )
}

function ExpandedDetails({
  entry,
  appDisplay,
}: {
  entry: TranscriptionEntry
  appDisplay: string | null
}) {
  const wpm = computeWpm(entry.wordCount, entry.speechDurationMs)
  const processingMs = Math.max(0, entry.durationMs - entry.speechDurationMs)
  const hasSpeechTiming = entry.speechDurationMs > 0
  const hasProcessing = entry.durationMs > 0

  const rows: { label: string; value: string }[] = []
  rows.push({ label: 'Date', value: formatFullTimestamp(entry.timestamp) })
  if (hasSpeechTiming) {
    const speech = formatPreciseDuration(entry.speechDurationMs)
    rows.push({
      label: 'Speech',
      value: wpm !== null ? `${speech} · ${wpm} wpm` : speech,
    })
  }
  if (hasProcessing) {
    rows.push({
      label: entry.wasCleanedUp ? 'Polish + processing' : 'Processing',
      value: formatPreciseDuration(processingMs),
    })
    rows.push({
      label: 'Total',
      value: formatPreciseDuration(entry.durationMs),
    })
  }
  rows.push({
    label: 'Words',
    value: `${entry.wordCount.toLocaleString()}`,
  })
  if (entry.targetApp) {
    rows.push({
      label: 'App',
      value: appDisplay
        ? appDisplay !== entry.targetApp
          ? `${appDisplay} · ${entry.targetApp}`
          : entry.targetApp
        : entry.targetApp,
    })
  }

  return (
    <dl className="mb-3 grid grid-cols-[110px_1fr] gap-x-4 gap-y-1.5 font-geist text-[11px]">
      {rows.map((row) => (
        <div key={row.label} className="contents">
          <dt className="font-medium uppercase tracking-[0.16em] text-white/40">
            {row.label}
          </dt>
          <dd
            className="text-white/80"
            style={{ fontFeatureSettings: '"tnum"' }}
          >
            {row.value}
          </dd>
        </div>
      ))}
    </dl>
  )
}
