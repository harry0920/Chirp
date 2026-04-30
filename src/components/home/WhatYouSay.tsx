interface Props {
  totalWords: number
  totalSessions: number
  longestDictationWords: number
  totalDurationMs: number
}

function formatDuration(ms: number): string {
  if (ms <= 0) return '0s'
  const totalSeconds = Math.round(ms / 1000)
  const h = Math.floor(totalSeconds / 3600)
  const m = Math.floor((totalSeconds % 3600) / 60)
  const s = totalSeconds % 60
  if (h > 0) return `${h}h ${m}m`
  if (m > 0) return `${m}m ${s}s`
  return `${s}s`
}

export function WhatYouSay({
  totalWords,
  totalSessions,
  longestDictationWords,
  totalDurationMs,
}: Props) {
  const avgWords = totalSessions > 0 ? Math.round(totalWords / totalSessions) : 0

  return (
    <article className="rounded-2xl border border-white/[0.06] bg-white/[0.02] p-5">
      <header className="mb-4 flex items-center justify-between">
        <span className="font-geist text-[10px] font-medium uppercase tracking-[0.2em] text-white/45">
          What you say
        </span>
      </header>

      <ul className="flex flex-col gap-3 font-geist text-[12px]">
        <Stat label="Avg per session" value={avgWords > 0 ? `${avgWords.toLocaleString()} words` : '—'} />
        <Stat
          label="Longest dictation"
          value={longestDictationWords > 0 ? `${longestDictationWords.toLocaleString()} words` : '—'}
        />
        <Stat label="Time spoken" value={totalDurationMs > 0 ? formatDuration(totalDurationMs) : '—'} />
      </ul>
    </article>
  )
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <li className="flex items-baseline justify-between">
      <span className="text-white/55">{label}</span>
      <span className="font-geist-mono text-[12px] text-white/85" style={{ fontFeatureSettings: '"tnum"' }}>
        {value}
      </span>
    </li>
  )
}
