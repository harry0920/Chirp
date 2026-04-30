interface Props {
  hourlyGrid: number[][]
}

const DAYS = ['M', 'T', 'W', 'T', 'F', 'S', 'S']

export function WhenYouDictate({ hourlyGrid }: Props) {
  const max = Math.max(1, ...hourlyGrid.flat())

  return (
    <article className="rounded-2xl border border-white/[0.06] bg-white/[0.02] p-5">
      <header className="mb-4 flex items-center justify-between">
        <span className="font-geist text-[10px] font-medium uppercase tracking-[0.2em] text-white/45">
          When you dictate
        </span>
      </header>

      <div className="flex flex-col gap-1.5">
        {hourlyGrid.map((row, day) => (
          <div key={day} className="flex items-center gap-2">
            <span className="w-3 font-geist-mono text-[9px] text-white/30">{DAYS[day]}</span>
            <div className="flex flex-1 items-center gap-[2px]">
              {row.map((count, hour) => {
                const intensity = count > 0 ? 0.18 + (count / max) * 0.82 : 0
                return (
                  <div
                    key={hour}
                    className="h-3 flex-1 rounded-[1px]"
                    style={{
                      backgroundColor:
                        count > 0
                          ? `rgba(255, 255, 255, ${intensity.toFixed(2)})`
                          : 'rgba(255, 255, 255, 0.04)',
                    }}
                    aria-label={`${DAYS[day]} ${hour}:00 — ${count} sessions`}
                  />
                )
              })}
            </div>
          </div>
        ))}
      </div>
    </article>
  )
}
