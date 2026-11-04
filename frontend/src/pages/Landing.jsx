import HeroSection from '../components/landing/HeroSection.jsx'
import HowItWorks from '../components/landing/HowItWorks.jsx'

export default function Landing() {
  return (
    <main>
      <HeroSection />
      <HowItWorks />
      <footer
        className="py-10 px-8 md:px-16 border-t"
        style={{ borderColor: '#A8B0BE', background: '#F5F4F0' }}
      >
        <p className="font-mono text-[10px] uppercase tracking-widest" style={{ color: '#A8B0BE' }}>
          ProsodyLab — Speech Prosody Analysis
        </p>
      </footer>
    </main>
  )
}
