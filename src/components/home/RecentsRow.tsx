import { useState } from 'react'
import { Copy, Trash2 } from 'lucide-react'
import type { TranscriptionEntry } from '../../stores/appStore'

interface Props {
  entries: TranscriptionEntry[]
  onCopy: (entry: TranscriptionEntry) => void
  onDelete: (entry: TranscriptionEntry) => void
  onViewAll: () => void
  resolveAppDisplay: (raw: string) => string
}

function formatTime(timestamp: string): string {
  const d = new Date(timestamp)
  return d.toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' })
    .replace(' AM', 'a')
    .replace(' PM', 'p')
    .toLowerCase()
}

function formatDuration(ms: number): string {
  const seconds = Math.round(ms / 1000)
  if (seconds < 60) return `${seconds}s`
  const m = Math.floor(seconds / 60)
  const s = seconds % 60
  return `${m}m ${s}s`
}

export function RecentsRow({ entries, onCopy, onDelete, onViewAll, resolveAppDisplay }: Props) {
  const [copiedId, setCopiedId] = useState<string | null>(null)

  return (
    <section>
      <header className="mb-3 flex items-center justify-between px-1">
        <span className="font-geist text-[10px] font-medium uppercase tracking-[0.2em] text-white/45">
          Recent
        </span>
        <button
          type="button"
          onClick={onViewAll}
          className="font-geist text-[11px] text-white/55 transition-colors hover:text-white"
        >
          View all →
        </button>
      </header>

      <ul className="card-surface divide-y divide-white/[0.06] px-5">
        {entries.map((entry) => {
          const appDisplay = entry.targetApp ? resolveAppDisplay(entry.targetApp) : null
          const id = entry.timestamp
          const copied = copiedId === id
          return (
            <li
              key={id}
              className="group grid grid-cols-[64px_1fr_auto] items-start gap-4 py-3"
            >
              <span
                className="pt-0.5 font-geist-mono text-[11px] text-white/35"
                style={{ fontFeatureSettings: '"tnum"' }}
              >
                {formatTime(entry.timestamp)}
              </span>
              <div className="min-w-0 flex flex-col gap-1">
                <p className="truncate font-geist text-[13px] text-white/85">{entry.text}</p>
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
                  onClick={() => {
                    onCopy(entry)
                    setCopiedId(id)
                    setTimeout(() => setCopiedId((c) => (c === id ? null : c)), 1200)
                  }}
                  aria-label="Copy text"
                  className="flex h-7 w-7 items-center justify-center rounded-full text-white/45 transition-colors hover:bg-white/[0.06] hover:text-white"
                >
                  <Copy size={13} />
                </button>
                <button
                  type="button"
                  onClick={() => onDelete(entry)}
                  aria-label="Delete entry"
                  className="flex h-7 w-7 items-center justify-center rounded-full text-white/45 transition-colors hover:bg-red-500/10 hover:text-red-400"
                >
                  <Trash2 size={13} />
                </button>
                {copied && (
                  <span className="font-geist-mono text-[10px] text-chirp-yellow">copied</span>
                )}
              </div>
            </li>
          )
        })}
      </ul>
    </section>
  )
}
