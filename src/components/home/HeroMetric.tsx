import { useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'
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

const PERIOD_NOUN: Record<Period, string> = {
  week: 'this week',
  month: 'this month',
  year: 'this year',
  all: 'lifetime',
}

/** Count animation from previous value to new target on every change.
 *  First mount counts up from 0 so the hero has motion on entrance. */
function useAnimatedNumber(target: number, duration = 700): number {
  const [value, setValue] = useState(0)
  const prevRef = useRef(0)
  const rafRef = useRef<number>(0)

  useEffect(() => {
    const from = prevRef.current
    const to = target
    if (from === to) return

    const start = performance.now()
    const animate = (now: number) => {
      const t = Math.min((now - start) / duration, 1)
      const eased = 1 - Math.pow(1 - t, 3) // ease-out cubic
      const v = Math.round(from + (to - from) * eased)
      setValue(v)
      if (t < 1) {
        rafRef.current = requestAnimationFrame(animate)
      } else {
        prevRef.current = to
      }
    }
    rafRef.current = requestAnimationFrame(animate)
    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current)
      prevRef.current = target
    }
  }, [target, duration])

  return value
}

export function HeroMetric({ daily, totalWords, period, onPeriodChange }: Props) {
  const sparklineData = useMemo(() => daily.map((d) => d.words), [daily])
  const hasData = sparklineData.some((v) => v > 0)
  const animatedTotal = useAnimatedNumber(totalWords)

  return (
    <section className="card-surface halo-hero relative overflow-hidden">
      <div className="px-8 pt-7">
        <div className="flex justify-end">
          <PeriodToggle period={period} onChange={onPeriodChange} />
        </div>
        <div className="mt-2 flex items-end gap-5">
          <span
            className="block font-geist font-semibold leading-none text-white"
            style={{
              fontFeatureSettings: '"tnum"',
              letterSpacing: '-0.04em',
              fontSize: 'clamp(64px, 9vw, 112px)',
            }}
          >
            {animatedTotal.toLocaleString()}
          </span>
          <span
            key={period}
            className="mb-3 block font-geist text-[11px] font-medium uppercase tracking-[0.18em] text-white/45 animate-fade-in"
          >
            Words {PERIOD_NOUN[period]}
          </span>
        </div>
      </div>

      <div className="mt-6 h-[64px] w-full px-1">
        {hasData ? (
          <Sparkline
            key={period}
            data={sparklineData}
            strokeWidth={1.25}
            dotRadius={2.75}
            endRatio={2 / 3}
            className="h-full w-full"
          />
        ) : (
          <div className="flex h-full w-full items-end px-7">
            <div className="h-px w-full bg-white/10" />
          </div>
        )}
      </div>
    </section>
  )
}

function PeriodToggle({ period, onChange }: { period: Period; onChange: (p: Period) => void }) {
  const containerRef = useRef<HTMLDivElement>(null)
  const buttonRefs = useRef<Array<HTMLButtonElement | null>>([])
  const [underline, setUnderline] = useState<{ left: number; width: number } | null>(null)

  // Re-measure the active button on period change AND on container resize.
  useLayoutEffect(() => {
    const measure = () => {
      const idx = PERIODS.findIndex((p) => p.id === period)
      const btn = buttonRefs.current[idx]
      const container = containerRef.current
      if (!btn || !container) return
      const containerRect = container.getBoundingClientRect()
      const btnRect = btn.getBoundingClientRect()
      setUnderline({
        left: btnRect.left - containerRect.left,
        width: btnRect.width,
      })
    }
    measure()
    if (!containerRef.current) return
    const ro = new ResizeObserver(measure)
    ro.observe(containerRef.current)
    return () => ro.disconnect()
  }, [period])

  return (
    <div
      ref={containerRef}
      className="relative flex items-center gap-1 font-geist text-[11px] uppercase tracking-[0.16em]"
    >
      {PERIODS.map((p, i) => (
        <span key={p.id} className="flex items-center gap-1">
          <button
            ref={(el) => { buttonRefs.current[i] = el }}
            type="button"
            onClick={() => onChange(p.id)}
            className={`relative transition-all duration-150 active:scale-95 ${
              p.id === period ? 'text-white' : 'text-white/40 hover:text-white/70'
            }`}
          >
            {p.label}
          </button>
          {i < PERIODS.length - 1 && <span className="text-white/20">·</span>}
        </span>
      ))}
      {underline && (
        <span
          aria-hidden
          className="pointer-events-none absolute -bottom-1 h-px bg-chirp-yellow"
          style={{
            left: `${underline.left}px`,
            width: `${underline.width}px`,
            transition:
              'left 280ms cubic-bezier(0.34, 1.56, 0.64, 1), width 280ms cubic-bezier(0.34, 1.56, 0.64, 1)',
          }}
        />
      )}
    </div>
  )
}
