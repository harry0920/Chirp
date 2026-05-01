import { useEffect } from 'react'
import { listen } from '@tauri-apps/api/event'
import { getCurrentWindow } from '@tauri-apps/api/window'
import { check } from '@tauri-apps/plugin-updater'
import { Check, Minus, Square, X } from 'lucide-react'
import { useAppStore } from '../../stores/appStore'
import { useTauri } from '../../hooks/useTauri'
import { AboutModal } from '../shared/AboutModal'
import { UpgradeModal } from '../shared/UpgradeModal'
import { Dock } from '../dock/Dock'
import { HomePage } from './HomePage'
import { VocabularyPage } from './VocabularyPage'
import { SnippetsPage } from './SnippetsPage'
import { ProPage } from './ProPage'
import { SettingsPage } from './SettingsPage'

const PAGES: Record<string, React.FC> = {
  home: HomePage,
  vocabulary: VocabularyPage,
  snippets: SnippetsPage,
  pro: ProPage,
  settings: SettingsPage,
}

const IS_MAC = navigator.platform.includes('Mac')

export function Settings() {
  const settingsPage = useAppStore((s) => s.settingsPage)
  const settingsSaved = useAppStore((s) => s.settingsSaved)
  const setSettingsSaved = useAppStore((s) => s.setSettingsSaved)
  const aboutModalOpen = useAppStore((s) => s.aboutModalOpen)
  const setAboutModalOpen = useAppStore((s) => s.setAboutModalOpen)
  const setUpgradeModalOpen = useAppStore((s) => s.setUpgradeModalOpen)
  const setUpdateAvailable = useAppStore((s) => s.setUpdateAvailable)
  const aiCleanup = useAppStore((s) => s.aiCleanup)
  const cleanupProvider = useAppStore((s) => s.cleanupProvider)
  const settingsLoaded = useAppStore((s) => s.settingsLoaded)
  const tauri = useTauri()

  useEffect(() => {
    if (!settingsLoaded || !aiCleanup || cleanupProvider !== 'local') return
    tauri.getLlmStatus().then((status) => {
      if (!status.modelDownloaded) {
        setUpgradeModalOpen(true)
      }
    }).catch(() => {})
  }, [aiCleanup, cleanupProvider, settingsLoaded, setUpgradeModalOpen, tauri])

  useEffect(() => {
    check().then((update) => {
      if (update) setUpdateAvailable(update.version)
    }).catch(() => { /* network error — silently ignore */ })
  }, [setUpdateAvailable])

  useEffect(() => {
    if (settingsSaved) {
      const timer = setTimeout(() => setSettingsSaved(false), 1500)
      return () => clearTimeout(timer)
    }
  }, [settingsSaved, setSettingsSaved])

  useEffect(() => {
    const unlisten = listen('check-for-updates', () => {
      setAboutModalOpen(true)
    })
    return () => { unlisten.then((f) => f()) }
  }, [setAboutModalOpen])

  const PageComponent = PAGES[settingsPage] ?? HomePage

  return (
    <div className="theme-pitch flex h-screen flex-col overflow-hidden bg-black no-select font-geist text-white">
      {/* Custom titlebar — drag region + Windows window controls */}
      <div
        data-tauri-drag-region
        className={`flex w-full shrink-0 items-center justify-end bg-black ${IS_MAC ? 'h-[34px]' : 'h-10'}`}
        style={IS_MAC ? { WebkitAppRegion: 'drag' } as React.CSSProperties : undefined}
      >
        {!IS_MAC && (
          <div className="flex h-full items-stretch">
            <button
              type="button"
              onClick={() => getCurrentWindow().minimize()}
              className="flex w-[46px] items-center justify-center text-white/40 transition-colors hover:bg-white/[0.04] hover:text-white/70"
              aria-label="Minimize"
            >
              <Minus size={16} />
            </button>
            <button
              type="button"
              onClick={() => getCurrentWindow().toggleMaximize()}
              className="flex w-[46px] items-center justify-center text-white/40 transition-colors hover:bg-white/[0.04] hover:text-white/70"
              aria-label="Maximize"
            >
              <Square size={12} />
            </button>
            <button
              type="button"
              onClick={() => getCurrentWindow().close()}
              className="flex w-[46px] items-center justify-center text-white/40 transition-colors hover:bg-red-500 hover:text-white"
              aria-label="Close"
            >
              <X size={16} />
            </button>
          </div>
        )}
      </div>

      {/* Full-bleed content with consistent gutters across every page */}
      <main className="relative flex-1 overflow-y-auto overflow-x-hidden">
        <div
          key={settingsPage}
          className="animate-fade-in mx-auto w-full max-w-[1080px] px-12 pt-10 pb-32"
        >
          <PageComponent />
        </div>
      </main>

      {/* Floating dock — bottom center */}
      <Dock />

      {/* Saved indicator */}
      {settingsSaved && (
        <div className="fixed bottom-24 right-6 z-50 flex items-center gap-1.5 rounded-full border border-white/10 bg-black/80 px-4 py-2 text-xs font-medium text-white shadow-elevated backdrop-blur-xl animate-saved-pop">
          <Check size={14} className="text-chirp-success" />
          Saved
        </div>
      )}

      {aboutModalOpen && <AboutModal />}
      <UpgradeModal />
    </div>
  )
}
