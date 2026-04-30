import { useEffect, useState, useRef, useCallback } from 'react'
import { getCurrentWindow } from '@tauri-apps/api/window'
import { availableMonitors, primaryMonitor } from '@tauri-apps/api/window'
import type { Monitor } from '@tauri-apps/api/window'
import { listen } from '@tauri-apps/api/event'
import { LogicalPosition, LogicalSize } from '@tauri-apps/api/dpi'
import { useAppStore } from '../../stores/appStore'
import { useAudio } from '../../hooks/useAudio'
import { useRecording } from '../../hooks/useRecording'
import { useOverlaySync } from '../../hooks/useOverlaySync'
import { BirdMark } from '../shared/BirdMark'
import { TransientCanvas } from './TransientCanvas'

const WIN_W = 300
const WIN_H = 64
const OFFSET = 80

const SNAP_RADIUS = 40

const SNAP_GRID = [
  { fx: 0.1, fy: 0, label: 'Top left' },
  { fx: 0.5, fy: 0, label: 'Top center' },
  { fx: 0.9, fy: 0, label: 'Top right' },
  { fx: 0.1, fy: 0.5, label: 'Middle left' },
  { fx: 0.5, fy: 0.5, label: 'Center' },
  { fx: 0.9, fy: 0.5, label: 'Middle right' },
  { fx: 0.1, fy: 1, label: 'Bottom left' },
  { fx: 0.5, fy: 1, label: 'Bottom center' },
  { fx: 0.9, fy: 1, label: 'Bottom right' },
] as const

function monitorLogicalBounds(m: Monitor) {
  const sf = m.scaleFactor
  return {
    x: m.position.x / sf,
    y: m.position.y / sf,
    w: m.size.width / sf,
    h: m.size.height / sf,
  }
}

function monitorSnapPoints(m: Monitor) {
  const b = monitorLogicalBounds(m)
  return SNAP_GRID.map((s) => ({
    x: b.x + (b.w - WIN_W) * s.fx,
    y: b.y + OFFSET + (b.h - WIN_H - OFFSET * 2) * s.fy,
    label: s.label,
  }))
}

function defaultPosition(primary: Monitor): { x: number; y: number } {
  const b = monitorLogicalBounds(primary)
  return {
    x: Math.round(b.x + (b.w - WIN_W) / 2),
    y: Math.round(b.y + b.h - OFFSET - WIN_H),
  }
}

async function resolvePosition(pos: unknown): Promise<{ x: number; y: number }> {
  const monitors = await availableMonitors()
  const primary = await primaryMonitor()
  const fallback = primary ? defaultPosition(primary) : { x: 100, y: 100 }

  if (typeof pos === 'string') {
    if (!primary) return fallback
    const b = monitorLogicalBounds(primary)
    if (pos === 'top') {
      return { x: Math.round(b.x + (b.w - WIN_W) / 2), y: Math.round(b.y + OFFSET) }
    }
    return fallback
  }

  if (pos && typeof pos === 'object' && 'x' in pos && 'y' in pos) {
    const { x, y } = pos as { x: number; y: number }
    const pillCX = x + WIN_W / 2
    const pillCY = y + WIN_H / 2
    for (const m of monitors) {
      const b = monitorLogicalBounds(m)
      if (pillCX >= b.x && pillCX < b.x + b.w && pillCY >= b.y && pillCY < b.y + b.h) {
        return { x, y }
      }
    }
    return fallback
  }

  return fallback
}

interface VirtualScreen {
  left: number
  top: number
  width: number
  height: number
  monitors: Monitor[]
}

function computeVirtualScreen(monitors: Monitor[]): VirtualScreen {
  let left = Infinity, top = Infinity, right = -Infinity, bottom = -Infinity
  for (const m of monitors) {
    const b = monitorLogicalBounds(m)
    left = Math.min(left, b.x)
    top = Math.min(top, b.y)
    right = Math.max(right, b.x + b.w)
    bottom = Math.max(bottom, b.y + b.h)
  }
  return { left, top, width: right - left, height: bottom - top, monitors }
}

type CanvasMode = 'listening' | 'polishing' | 'error'

export function Overlay() {
  const status = useAppStore((s) => s.status)
  const setStatus = useAppStore((s) => s.setStatus)
  const amplitudes = useAppStore((s) => s.amplitudes)
  const position = useAppStore((s) => s.overlayPosition)
  const updateSettings = useAppStore((s) => s.updateSettings)
  const [dismissing, setDismissing] = useState(false)

  const [repositioning, setRepositioning] = useState(false)
  const [dragPos, setDragPos] = useState<{ x: number; y: number } | null>(null)
  const [dropped, setDropped] = useState(false)
  const [dragging, setDragging] = useState(false)
  const [virtualScreen, setVirtualScreen] = useState<VirtualScreen | null>(null)
  const [snapPoints, setSnapPoints] = useState<Array<{ x: number; y: number; label: string }>>([])
  const draggingRef = useRef(false)
  const dragStartRef = useRef<{ mouseX: number; mouseY: number; pillX: number; pillY: number } | null>(null)
  const originalPositionRef = useRef<{ x: number; y: number } | null>(null)

  useOverlaySync()
  useAudio()
  useRecording()

  // Strictly transient: only render during the active arc
  // (listening → polishing → fade). Idle and the legacy passive
  // indicator do not produce any visual.
  const canvasMode: CanvasMode | null =
    status === 'listening'
      ? 'listening'
      : status === 'processing' || status === 'polishing' || status === 'done'
        ? 'polishing'
        : status === 'error'
          ? 'error'
          : null

  const shouldShow = canvasMode !== null

  useEffect(() => {
    const unlisten = listen('enter-reposition-mode', async () => {
      const win = getCurrentWindow()
      const monitors = await availableMonitors()
      if (monitors.length === 0) return

      const vs = computeVirtualScreen(monitors)
      const resolved = await resolvePosition(position)
      originalPositionRef.current = resolved

      const allSnaps: Array<{ x: number; y: number; label: string }> = []
      for (const m of monitors) {
        allSnaps.push(...monitorSnapPoints(m))
      }
      setSnapPoints(allSnaps)

      setDragPos({ x: resolved.x - vs.left, y: resolved.y - vs.top })
      setVirtualScreen(vs)
      setDropped(false)
      setRepositioning(true)

      await win.setIgnoreCursorEvents(false)
      await win.setSize(new LogicalSize(vs.width, vs.height))
      await win.setPosition(new LogicalPosition(vs.left, vs.top))
      await win.show()
    })

    return () => { unlisten.then((fn) => fn()) }
  }, [position])

  const exitReposition = useCallback(async (savePosition: boolean) => {
    if (!virtualScreen) return

    const win = getCurrentWindow()

    const finalPos = savePosition && dragPos
      ? {
          x: Math.round(dragPos.x + virtualScreen.left),
          y: Math.round(dragPos.y + virtualScreen.top),
        }
      : originalPositionRef.current || { x: 100, y: 100 }

    if (savePosition) {
      updateSettings({ overlayPosition: finalPos })
    }

    await win.setSize(new LogicalSize(WIN_W, WIN_H))
    await win.setPosition(new LogicalPosition(finalPos.x, finalPos.y))
    await win.setIgnoreCursorEvents(true)

    await win.emit('reposition-complete', finalPos)

    setRepositioning(false)
    setDragPos(null)
    setDropped(false)
    setDragging(false)
    setVirtualScreen(null)
    draggingRef.current = false
    dragStartRef.current = null
    originalPositionRef.current = null
    setSnapPoints([])
  }, [virtualScreen, dragPos, updateSettings])

  useEffect(() => {
    if (!repositioning) return
    const unlisten = listen('hotkey-pressed', () => { exitReposition(false) })
    return () => { unlisten.then((fn) => fn()) }
  }, [repositioning, exitReposition])

  useEffect(() => {
    if (!repositioning) return
    const handleKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        e.preventDefault()
        exitReposition(false)
      } else if (e.key === 'Enter') {
        e.preventDefault()
        exitReposition(true)
      }
    }
    window.addEventListener('keydown', handleKey)
    return () => window.removeEventListener('keydown', handleKey)
  }, [repositioning, exitReposition])

  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    if (!dragPos) return
    e.preventDefault()
    draggingRef.current = true
    setDragging(true)
    setDropped(false)
    dragStartRef.current = {
      mouseX: e.clientX,
      mouseY: e.clientY,
      pillX: dragPos.x,
      pillY: dragPos.y,
    }
  }, [dragPos])

  useEffect(() => {
    if (!repositioning || !virtualScreen) return

    const handleMouseMove = (e: MouseEvent) => {
      if (!draggingRef.current || !dragStartRef.current) return
      const dx = e.clientX - dragStartRef.current.mouseX
      const dy = e.clientY - dragStartRef.current.mouseY
      let newX = Math.max(0, Math.min(virtualScreen.width - WIN_W, dragStartRef.current.pillX + dx))
      let newY = Math.max(0, Math.min(virtualScreen.height - WIN_H, dragStartRef.current.pillY + dy))

      const absCX = newX + virtualScreen.left + WIN_W / 2
      const absCY = newY + virtualScreen.top + WIN_H / 2
      for (const snap of snapPoints) {
        const sx = snap.x + WIN_W / 2
        const sy = snap.y + WIN_H / 2
        const dist = Math.sqrt((absCX - sx) ** 2 + (absCY - sy) ** 2)
        if (dist < SNAP_RADIUS) {
          newX = snap.x - virtualScreen.left
          newY = snap.y - virtualScreen.top
          break
        }
      }

      setDragPos({ x: newX, y: newY })
    }

    const handleMouseUp = () => {
      if (!draggingRef.current) return
      draggingRef.current = false
      setDragging(false)
      dragStartRef.current = null
      setDropped(true)
    }

    window.addEventListener('mousemove', handleMouseMove)
    window.addEventListener('mouseup', handleMouseUp)
    return () => {
      window.removeEventListener('mousemove', handleMouseMove)
      window.removeEventListener('mouseup', handleMouseUp)
    }
  }, [repositioning, virtualScreen, snapPoints])

  useEffect(() => {
    if (repositioning) return
    const win = getCurrentWindow()
    resolvePosition(position).then(async (pos) => {
      await win.setSize(new LogicalSize(WIN_W, WIN_H))
      await win.setPosition(new LogicalPosition(pos.x, pos.y))
      await win.setIgnoreCursorEvents(true)
    })
  }, [position, repositioning])

  useEffect(() => {
    if (repositioning) return
    const win = getCurrentWindow()
    if (shouldShow) {
      win.show()
    } else {
      win.hide()
    }
  }, [shouldShow, repositioning])

  // Auto-dismiss arc: done → fade out (200ms), error → flash → fade
  useEffect(() => {
    const delay = status === 'done' ? 250 : status === 'error' ? 1500 : null
    if (delay === null) return

    const timer = setTimeout(() => {
      setDismissing(true)
      setTimeout(() => {
        setStatus('idle')
        setDismissing(false)
      }, 220)
    }, delay)
    return () => clearTimeout(timer)
  }, [status, setStatus])

  if (repositioning && dragPos && virtualScreen) {
    const snapDots = snapPoints.map((s) => ({
      x: s.x - virtualScreen.left + WIN_W / 2,
      y: s.y - virtualScreen.top + WIN_H / 2,
      label: s.label,
      absX: s.x,
      absY: s.y,
    }))

    return (
      <div className="fixed inset-0 bg-black/70">
        <div className="absolute left-0 right-0 top-8 flex items-start justify-center gap-4">
          <div className="flex flex-col items-center gap-2">
            <div className="rounded-full border border-white/10 bg-black/60 px-4 py-2 backdrop-blur-md">
              <span className="font-geist text-[13px] font-medium text-white/90">
                Drag the pill to reposition
              </span>
            </div>
            <span className="font-geist text-[11px] uppercase tracking-[0.16em] text-white/40">
              Enter to confirm · Escape to cancel
            </span>
          </div>
          {dropped && (
            <button
              onClick={() => exitReposition(true)}
              className="animate-fade-in rounded-full bg-white px-5 py-2 font-geist text-[13px] font-semibold text-black transition-transform hover:-translate-y-px active:translate-y-0"
            >
              Done
            </button>
          )}
        </div>

        {snapDots.map((dot, i) => (
          <button
            key={i}
            className="group absolute flex flex-col items-center gap-1.5 -translate-x-1/2 -translate-y-1/2"
            style={{ left: dot.x, top: dot.y }}
            onClick={() => {
              setDragPos({ x: dot.absX - virtualScreen.left, y: dot.absY - virtualScreen.top })
              setDropped(true)
            }}
          >
            <div className="h-3 w-3 rounded-full border border-white/30 bg-white/[0.06] transition-all group-hover:scale-150 group-hover:border-white/70 group-hover:bg-white/20" />
            <span className="font-geist text-[10px] text-white/0 transition-colors group-hover:text-white/55">
              {dot.label}
            </span>
          </button>
        ))}

        <div
          style={{
            position: 'absolute',
            left: dragPos.x,
            top: dragPos.y,
            width: WIN_W,
            height: WIN_H,
            cursor: dragging ? 'grabbing' : 'grab',
          }}
          className="flex items-center justify-center"
          onMouseDown={handleMouseDown}
        >
          <div className="flex h-9 items-center gap-2.5 rounded-full border border-white/10 bg-black/70 px-3.5 backdrop-blur-xl">
            <BirdMark size={16} color="#FFFFFF" />
            <span className="font-geist text-[12px] font-medium text-white/85 select-none">Chirp</span>
          </div>
        </div>
      </div>
    )
  }

  if (!shouldShow || !canvasMode) return null

  return (
    <div className="pointer-events-none flex h-screen w-screen items-center justify-center">
      <TransientCanvas mode={canvasMode} amplitudes={amplitudes} dismissing={dismissing} />
    </div>
  )
}
