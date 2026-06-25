import { useEffect, useRef, useState } from "react";
import { Waves, Mic, BrainCircuit, Activity, Languages, ChevronRight, FileAudio2, Shield, Zap } from "lucide-react";
import "./LandingPage.css";

const FEATURES = [
  {
    icon: Mic,
    color: "#22d3ee",
    title: "Live Audio Capture",
    desc: "Record directly from your microphone or upload any audio file. Supports WAV, MP3, M4A, FLAC and more."
  },
  {
    icon: BrainCircuit,
    color: "#818cf8",
    title: "AI Speech Diagnostics",
    desc: "Ensemble of Random Forest, XGBoost & SVM classifiers extract 61 acoustic features to classify age, gender and speech typicality."
  },
  {
    icon: Languages,
    color: "#f472b6",
    title: "Bilingual Transcription",
    desc: "Simultaneous Kannada (ಕನ್ನಡ) and English transcription powered by faster-whisper — side-by-side results in seconds."
  },
  {
    icon: Activity,
    color: "#34d399",
    title: "Clinical Analysis",
    desc: "Pitch, jitter, shimmer, HNR, speech rate and more — every biomarker referenced against clinical norms."
  },
  {
    icon: Zap,
    color: "#fbbf24",
    title: "Explainable AI",
    desc: "SHAP feature drivers reveal exactly which acoustic signals pushed each prediction, not just a black-box result."
  },
  {
    icon: FileAudio2,
    color: "#2dd4bf",
    title: "PDF Report Export",
    desc: "One-click professional diagnostic report with all metrics, charts and transcripts, ready to share with clinicians."
  }
];

const STATS = [
  { value: "61", label: "Acoustic Features" },
  { value: "3", label: "ML Classifiers" },
  { value: "2", label: "Languages" },
  { value: "<30s", label: "Inference Time" }
];

/* Animated audio bars for decoration */
function AudioBars({ count = 32, color = "#22d3ee" }) {
  return (
    <div className="lp-audiobars">
      {Array.from({ length: count }).map((_, i) => (
        <div
          key={i}
          className="lp-bar"
          style={{
            background: color,
            animationDelay: `${(i * 0.07).toFixed(2)}s`,
            animationDuration: `${0.8 + (i % 5) * 0.15}s`
          }}
        />
      ))}
    </div>
  );
}

/* Floating particle orbs */
function Orbs() {
  return (
    <div className="lp-orbs" aria-hidden="true">
      <div className="lp-orb lp-orb-1" />
      <div className="lp-orb lp-orb-2" />
      <div className="lp-orb lp-orb-3" />
    </div>
  );
}

export default function LandingPage({ onEnter }) {
  const [visible, setVisible] = useState(false);
  const [cardVisible, setCardVisible] = useState([]);
  const [hoveredCard, setHoveredCard] = useState(null);

  useEffect(() => {
    const t1 = setTimeout(() => setVisible(true), 80);
    const timers = FEATURES.map((_, i) =>
      setTimeout(() => setCardVisible(prev => [...prev, i]), 300 + i * 110)
    );
    return () => { clearTimeout(t1); timers.forEach(clearTimeout); };
  }, []);

  const handleEnter = () => {
    // Fade out then call onEnter
    document.getElementById("lp-root")?.classList.add("lp-exit");
    setTimeout(onEnter, 420);
  };

  return (
    <div id="lp-root" className={`lp-root ${visible ? "lp-visible" : ""}`}>
      <Orbs />

      {/* ── HERO ── */}
      <section className="lp-hero">
        <div className="lp-hero-inner">
          {/* Logo badge */}
          <div className="lp-logo-ring">
            <div className="lp-logo-inner">
              <Waves size={32} color="#fff" strokeWidth={2.2} />
            </div>
            <div className="lp-logo-pulse" />
            <div className="lp-logo-pulse lp-logo-pulse-2" />
          </div>

          <div className="lp-pill">
            <span className="lp-pill-dot" />
            AI-Powered · Kannada + English · Clinical Grade
          </div>

          <h1 className="lp-headline">
            Speak<span className="lp-accent">Sense</span>
          </h1>
          <p className="lp-sub">
            Automated Speech Diagnostic Engine
          </p>
          <p className="lp-desc">
            Upload or record any audio — SpeakSense extracts 61 acoustic biomarkers,
            classifies speech typicality with ensemble ML, transcribes in Kannada and
            English, and delivers a clinical-grade PDF report in under 30 seconds.
          </p>

          {/* Audio bars decoration */}
          <AudioBars count={28} color="#22d3ee" />

          <button
            id="lp-enter-btn"
            className="lp-cta"
            onClick={handleEnter}
            aria-label="Open SpeakSense Diagnostics Dashboard"
          >
            <span>Launch Diagnostics</span>
            <ChevronRight size={20} strokeWidth={2.5} />
          </button>

          <p className="lp-tagline">
            AblePro × BIG Foundation Hackathon 2026
          </p>
        </div>
      </section>

      {/* ── STATS STRIP ── */}
      <section className="lp-stats-strip">
        {STATS.map(({ value, label }) => (
          <div key={label} className="lp-stat">
            <span className="lp-stat-val">{value}</span>
            <span className="lp-stat-label">{label}</span>
          </div>
        ))}
      </section>

      {/* ── FEATURES GRID ── */}
      <section className="lp-features">
        <div className="lp-section-header">
          <h2 className="lp-section-title">Everything you need in one engine</h2>
          <p className="lp-section-sub">From raw audio to clinical insights — end to end, in seconds.</p>
        </div>

        <div className="lp-grid">
          {FEATURES.map(({ icon: Icon, color, title, desc }, i) => (
            <div
              key={title}
              className={`lp-card ${cardVisible.includes(i) ? "lp-card-in" : ""}`}
              onMouseEnter={() => setHoveredCard(i)}
              onMouseLeave={() => setHoveredCard(null)}
              style={{ "--card-accent": color }}
            >
              <div className="lp-card-icon" style={{ background: `${color}18`, border: `1px solid ${color}30` }}>
                <Icon size={22} color={color} strokeWidth={1.8} />
              </div>
              <div className="lp-card-body">
                <h3 className="lp-card-title">{title}</h3>
                <p className="lp-card-desc">{desc}</p>
              </div>
              <div className="lp-card-glow" style={{ background: `radial-gradient(circle at 50% 0%, ${color}14 0%, transparent 70%)` }} />
            </div>
          ))}
        </div>
      </section>

      {/* ── BOTTOM CTA ── */}
      <section className="lp-bottom-cta">
        <div className="lp-bottom-inner">
          <Shield size={20} color="#818cf8" />
          <p>
            SpeakSense is designed for assistive diagnostics.
            Results are intended to support, not replace, clinical evaluation.
          </p>
          <button className="lp-cta lp-cta-sm" onClick={handleEnter} id="lp-bottom-enter-btn">
            <span>Open Dashboard</span>
            <ChevronRight size={16} strokeWidth={2.5} />
          </button>
        </div>
      </section>

      {/* ── FOOTER ── */}
      <footer className="lp-footer">
        <Waves size={14} color="var(--muted)" />
        <span>SpeakSense · AblePro × BIG Foundation · 2026</span>
      </footer>
    </div>
  );
}
