import { useMemo } from 'react'
import type { DailyPoint, Period } from './types'
import { Sparkline } from '../shared/Sparkline'

interface Props {
  daily: DailyPoint[]
  totalWords: number
  period: Period
  onPeriodChange: (p: Period) => void
}

const PERIODS: { id: Period; label: string }[] = [
  { id: 'week', label: 'week' },
  { id: 'month', label: 'month' },
  { id: 'year', label: 'year' },
  { id: 'all', label: 'all' },
]

export function HeroMetric({ daily, totalWords, period, onPeriodChange }: Props) {
  const sparklineData = useMemo(() => daily.map((d) => d.words), [daily])
  const hasData = sparklineData.some((v) => v > 0)

  return (
    <section className="halo-hero relative">
      <div className="flex items-start justify-between">
        <span className="font-geist text-[11px] font-medium uppercase tracking-[0.18em] text-white/45">
          Words this {period === 'all' ? 'lifetime' : period}
        </span>
        <PeriodToggle period={period} onChange={onPeriodChange} />
      </div>

      <div
        className="mt-3 font-geist text-[112px] font-semibold leading-none text-white"
        style={{ fontFeatureSettings: '"tnum"', letterSpacing: '-0.04em' }}
      >
        {totalWords.toLocaleString()}
      </div>

      <div className="mt-6">
        {hasData ? (
          <Sparkline
            data={sparklineData}
            width={720}
            height={56}
            strokeWidth={1.25}
            dotRadius={2.75}
            className="w-full"
          />
        ) : (
          <div className="h-[56px] w-full border-t border-dashed border-white/10" />
        )}
      </div>
    </section>
  )
}

function PeriodToggle({ period, onChange }: { period: Period; onChange: (p: Period) => void }) {
  return (
    <div className="flex items-center gap-1 font-geist text-[11px] uppercase tracking-[0.16em]">
      {PERIODS.map((p, i) => (
        <span key={p.id} className="flex items-center gap-1">
          <button
            type="button"
            onClick={() => onChange(p.id)}
            className={`transition-colors ${
              p.id === period ? 'text-white' : 'text-white/40 hover:text-white/70'
            }`}
          >
            {p.label}
          </button>
          {i < PERIODS.length - 1 && <span className="text-white/20">·</span>}
        </span>
      ))}
    </div>
  )
}
