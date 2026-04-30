interface SparklineProps {
  data: number[]
  /** SVG viewBox aspect — the rendered SVG fills its parent regardless,
   *  this just controls the drawing coordinate system. */
  viewWidth?: number
  viewHeight?: number
  strokeWidth?: number
  dotRadius?: number
  lineColor?: string
  dotColor?: string
  showDot?: boolean
  padding?: number
  className?: string
}

export function Sparkline({
  data,
  viewWidth = 1000,
  viewHeight = 100,
  strokeWidth = 1.25,
  dotRadius = 2.5,
  lineColor = '#FFFFFF',
  dotColor = '#F0B723',
  showDot = true,
  padding = 4,
  className = '',
}: SparklineProps) {
  if (data.length === 0) {
    return (
      <svg
        viewBox={`0 0 ${viewWidth} ${viewHeight}`}
        preserveAspectRatio="none"
        className={className}
        style={{ width: '100%', height: '100%' }}
        xmlns="http://www.w3.org/2000/svg"
      />
    )
  }

  const innerH = viewHeight - padding * 2
  const innerW = viewWidth
  const min = Math.min(...data)
  const max = Math.max(...data)
  const range = max - min || 1

  const stepX = data.length > 1 ? innerW / (data.length - 1) : 0
  const points = data.map((v, i) => {
    const x = i * stepX
    const y = padding + innerH - ((v - min) / range) * innerH
    return [x, y] as const
  })

  const polyline = points.map(([x, y]) => `${x.toFixed(2)},${y.toFixed(2)}`).join(' ')
  const last = points[points.length - 1]

  return (
    <svg
      viewBox={`0 0 ${viewWidth} ${viewHeight}`}
      preserveAspectRatio="none"
      className={className}
      style={{ width: '100%', height: '100%', overflow: 'visible' }}
      xmlns="http://www.w3.org/2000/svg"
      shapeRendering="geometricPrecision"
    >
      {data.length > 1 && (
        <polyline
          points={polyline}
          fill="none"
          stroke={lineColor}
          strokeWidth={strokeWidth}
          strokeLinejoin="miter"
          strokeLinecap="butt"
          vectorEffect="non-scaling-stroke"
        />
      )}
      {showDot && (
        <circle
          cx={last[0]}
          cy={last[1]}
          r={dotRadius}
          fill={dotColor}
          vectorEffect="non-scaling-stroke"
        />
      )}
    </svg>
  )
}
