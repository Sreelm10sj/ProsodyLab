import React from 'react'

const STEPS = [
  { id: 1, label: 'Dataset', description: 'Feature cache' },
  { id: 2, label: 'Anchors', description: 'Acoustic discovery' },
  { id: 3, label: 'Training', description: 'Iterative clustering' },
  { id: 4, label: 'Results', description: 'Bundle export' },
]

export default function TrainingStepper({
  currentStep,
  maxCompletedStep,
  onSelectStep,
  completedSteps = [],
  onStepClick,
}) {
  const handleSelect = onStepClick || onSelectStep

  return (
    <nav aria-label="Training stages" className="w-full">
      <ol className="grid grid-cols-2 md:grid-cols-4 gap-1.5 rounded-2xl border border-slate-200 bg-white p-1.5 shadow-2xs">
        {STEPS.map((step) => {
          const isActive = currentStep === step.id
          const isCompleted = completedSteps.includes(step.id) || currentStep > step.id
          const isClickable = (isCompleted || (maxCompletedStep != null && step.id <= maxCompletedStep + 1)) && !isActive

          return (
            <li
              key={step.id}
              onClick={() => isClickable && handleSelect && handleSelect(step.id)}
              className={`p-3 rounded-xl transition-all select-none ${
                isActive
                  ? 'bg-slate-900 text-white shadow-xs'
                  : isCompleted
                  ? 'hover:bg-slate-50 text-slate-700 hover:text-slate-900 cursor-pointer border border-transparent hover:border-slate-200/80'
                  : 'text-slate-400 opacity-60 cursor-not-allowed border border-transparent'
              }`}
              role={isClickable ? 'button' : undefined}
              tabIndex={isClickable ? 0 : -1}
              onKeyDown={(e) => {
                if (isClickable && (e.key === 'Enter' || e.key === ' ')) {
                  e.preventDefault()
                  handleSelect?.(step.id)
                }
              }}
            >
              <div className="flex items-center justify-between mb-1">
                <span
                  className={`font-mono text-[10px] font-bold uppercase tracking-wider ${
                    isActive ? 'text-amber-400' : isCompleted ? 'text-slate-400' : 'text-slate-400'
                  }`}
                >
                  0{step.id}
                </span>
                {isCompleted && (
                  <span className="inline-flex items-center justify-center w-4 h-4 rounded-full bg-emerald-50 border border-emerald-200 text-emerald-700 font-mono text-[10px] font-bold">
                    ✓
                  </span>
                )}
                {isActive && (
                  <span className="inline-block w-2 h-2 rounded-full bg-amber-400 animate-pulse" />
                )}
              </div>
              <div
                className={`font-sans font-semibold text-xs md:text-sm leading-tight ${
                  isActive ? 'text-white' : isCompleted ? 'text-slate-900' : 'text-slate-500'
                }`}
              >
                {step.label}
              </div>
              <div
                className={`font-mono text-[10px] truncate mt-0.5 ${
                  isActive ? 'text-slate-300' : 'text-slate-400'
                }`}
              >
                {step.description}
              </div>
            </li>
          )
        })}
      </ol>
    </nav>
  )
}
