import { useState } from 'react'
import { trackEvent } from '@aptabase/tauri'
import { useAppStore } from '../../stores/appStore'
import { BirdMark } from '../shared/BirdMark'
import { Welcome } from './Welcome'
import { SetupStep } from './SetupStep'
import { PermissionsStep } from './PermissionsStep'
import { ModelDownload } from './ModelDownload'
import { HelpImprove } from './HelpImprove'

const IS_MAC = navigator.platform.includes('Mac')
const STEPS = IS_MAC ? 5 : 4

export function Onboarding() {
  const [step, setStep] = useState(0)
  const setOnboardingComplete = useAppStore((s) => s.setOnboardingComplete)

  const handleFinish = () => {
    trackEvent('onboarding_completed', { steps_completed: String(STEPS) })
    setOnboardingComplete(true)
  }

  const modelStep = IS_MAC ? 3 : 2
  const helpStep = IS_MAC ? 4 : 3

  return (
    <div className="theme-pitch flex h-screen flex-col overflow-hidden bg-black font-geist text-white">
      <div className="mx-auto flex w-full max-w-[640px] flex-1 flex-col items-center justify-center px-12">
        <div className="halo-mark relative mb-3 flex items-center justify-center">
          <BirdMark size={56} color="#FFFFFF" />
        </div>
        <h1 className="font-geist text-[26px] font-semibold tracking-tight text-white">
          chirp
        </h1>

        <div className="mt-12 w-full">
          {step === 0 && <Welcome onNext={() => setStep(1)} />}
          {step === 1 && <SetupStep onNext={() => setStep(2)} />}
          {IS_MAC && step === 2 && <PermissionsStep onNext={() => setStep(3)} />}
          {step === modelStep && <ModelDownload onFinish={() => setStep(helpStep)} />}
          {step === helpStep && <HelpImprove onNext={handleFinish} />}
        </div>
      </div>

      <div className="flex shrink-0 items-center justify-center pb-10">
        <div className="flex items-center gap-2">
          {Array.from({ length: STEPS }, (_, i) => (
            <span
              key={i}
              className={`block h-1 rounded-full transition-all duration-300 ${
                i === step
                  ? 'w-6 bg-white'
                  : i < step
                    ? 'w-1 bg-white/55'
                    : 'w-1 bg-white/15'
              }`}
            />
          ))}
        </div>
      </div>
    </div>
  )
}
