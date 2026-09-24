import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  Activity,
  ArrowDown,
  ArrowRight,
  Crosshair,
  Maximize2,
  Radio,
  Scan,
  Shield,
  Sliders,
  Volume2,
  VolumeX,
  Waves,
  Zap,
} from 'lucide-react';
import { LandingScrollSections } from './LandingScrollSections';
import { getAssetUrl } from '../../utils/assets';
import './cinematic.css';

export interface CinematicLandingProps {
  onEnterMission: () => void;
  onExploreDemo?: (scenario: string) => void;
}

export const CinematicLanding: React.FC<CinematicLandingProps> = ({
  onEnterMission,
  onExploreDemo,
}) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [bootStep, setBootStep] = useState<number>(0);
  const [bootProgress, setBootProgress] = useState<number>(0);
  const [bootComplete, setBootComplete] = useState<boolean>(false);
  const [targetDetected, setTargetDetected] = useState<boolean>(false);
  const [videoError, setVideoError] = useState<boolean>(false);

  // 1. Opening Boot Sequence (Fast ~1.5s total)
  useEffect(() => {
    if (bootComplete) return;

    const t1 = setTimeout(() => {
      setBootStep(1);
      setBootProgress(30);
    }, 150); // SONAR ARRAY ONLINE

    const t2 = setTimeout(() => {
      setBootStep(2);
      setBootProgress(60);
    }, 550); // AI INFERENCE READY

    const t3 = setTimeout(() => {
      setBootStep(3);
      setBootProgress(85);
    }, 950); // TARGET ANALYSIS READY

    const t4 = setTimeout(() => {
      setBootStep(4);
      setBootProgress(100);
    }, 1250); // MISSION LINK ESTABLISHED

    const t5 = setTimeout(() => {
      setBootComplete(true);
      if (videoRef.current) {
        videoRef.current.play().catch(() => {});
      }
    }, 1550);

    return () => {
      clearTimeout(t1);
      clearTimeout(t2);
      clearTimeout(t3);
      clearTimeout(t4);
      clearTimeout(t5);
    };
  }, [bootComplete]);

  // Fast forward skip
  const handleSkipBoot = useCallback(() => {
    setBootStep(5);
    setBootProgress(100);
    setBootComplete(true);
    if (videoRef.current) {
      videoRef.current.play().catch(() => {});
    }
  }, []);

  // Listen for Escape key to fast forward
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && !bootComplete) {
        handleSkipBoot();
      }
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [bootComplete, handleSkipBoot]);

  // 2. Cinematic AI Detection Event Trigger (~3.5s after boot)
  useEffect(() => {
    if (!bootComplete) return;

    const targetTimer = setTimeout(() => {
      setTargetDetected(true);
    }, 3500);

    return () => clearTimeout(targetTimer);
  }, [bootComplete]);

  const handleScrollToExplore = () => {
    const el = document.getElementById('explore-system-section');
    if (el) {
      el.scrollIntoView({ behavior: 'smooth' });
    }
  };

  return (
    <div className="cinematic-experience-root">
      {/* ===================================================================
          1. OPENING SEQUENCE (Subtle Black Screen Boot)
          =================================================================== */}
      <div className={`cinematic-boot-screen ${bootComplete ? 'fade-out' : ''}`}>
        <div className="cinematic-boot-console">
          <div className="boot-header-brand">
            <Waves className="boot-logo-icon" />
            <div>
              <div className="boot-title">SONARIS-X</div>
              <div className="boot-subtitle">ACOUSTIC INTELLIGENCE SYSTEM</div>
            </div>
          </div>

          <div className="boot-lines-container">
            <div className={`boot-line ${bootStep >= 0 ? 'visible' : ''}`}>
              <span className="boot-label">INITIALIZING MISSION ENVIRONMENT</span>
              <span className="boot-dots" />
              <span className="boot-status ready">INITIALIZING...</span>
            </div>

            <div className={`boot-line ${bootStep >= 1 ? 'visible' : ''}`}>
              <span className="boot-label">SONAR ARRAY</span>
              <span className="boot-dots" />
              <span className="boot-status online">ONLINE</span>
            </div>

            <div className={`boot-line ${bootStep >= 2 ? 'visible' : ''}`}>
              <span className="boot-label">AI INFERENCE</span>
              <span className="boot-dots" />
              <span className="boot-status ready">READY</span>
            </div>

            <div className={`boot-line ${bootStep >= 3 ? 'visible' : ''}`}>
              <span className="boot-label">TARGET ANALYSIS</span>
              <span className="boot-dots" />
              <span className="boot-status ready">READY</span>
            </div>

            <div className={`boot-line ${bootStep >= 4 ? 'visible' : ''}`}>
              <span className="boot-label">MISSION LINK</span>
              <span className="boot-dots" />
              <span className="boot-status established">ESTABLISHED</span>
            </div>
          </div>

          <div className="boot-progress-bar-wrap">
            <div className="boot-progress-bar-fill" style={{ width: `${bootProgress}%` }} />
          </div>

          <div className="boot-footer-meta">
            <span>SECURE OCEAN INTELLIGENCE · v2.4</span>
            <button className="boot-skip-btn" onClick={handleSkipBoot}>
              FAST FORWARD [ESC]
            </button>
          </div>
        </div>
      </div>

      {/* ===================================================================
          2. FULL-SCREEN VIDEO HERO (100vw × 100vh)
          =================================================================== */}
      <section className="cinematic-video-hero">
        {/* Fullscreen Video Background */}
        <div className="hero-video-wrapper">
          {!videoError ? (
            <video
              ref={videoRef}
              autoPlay
              loop
              muted
              playsInline
              preload="metadata"
              poster={getAssetUrl('videos/sonaris-hero-poster.jpg')}
              className="hero-cinematic-video"
              onError={() => setVideoError(true)}
            >
              <source src={getAssetUrl('videos/sonaris-hero.mp4')} type="video/mp4" />
            </video>
          ) : (
            <img
              src={getAssetUrl('videos/sonaris-hero-poster.jpg')}
              alt="Underwater Sonar Mission Footage"
              className="hero-cinematic-video"
            />
          )}

          {/* Dark Cinematic Color Grade Overlay */}
          <div className="hero-video-overlay-gradient" />
          <div className="hero-corner-shade" />
          <div className="hero-video-scanlines" />

          {/* Horizontal Acoustic Sonar Scanning Sweep Line */}
          <div className="hero-acoustic-sweep-line" />

          {/* Sweep Status Indicator */}
          <div className="acoustic-sweep-alert">
            <Radio size={12} className="pulse-beacon" />
            <span>ACOUSTIC RETURN DETECTED // TARGET ANALYSIS</span>
          </div>
        </div>

        {/* Live Telemetry HUD Overlays */}
        {bootComplete && (
          <div className="cinematic-hud-overlay">
            {/* Top Telemetry Bar */}
            <div className="hud-top-bar">
              <div className="hud-mission-badge">
                <span className="hud-callsign">MISSION: SONAR-01</span>
                <span className="hud-coordinates">LAT 44°18.42' N · LON 068°12.91' W</span>
              </div>

              <div className="hud-telemetry-cluster">
                <div className="hud-telemetry-item">
                  <span className="hud-telemetry-label">DEPTH</span>
                  <span className="hud-telemetry-val">
                    42.8<em>m</em>
                  </span>
                </div>
                <div className="hud-telemetry-item">
                  <span className="hud-telemetry-label">ALTITUDE</span>
                  <span className="hud-telemetry-val">
                    8.4<em>m</em>
                  </span>
                </div>
                <div className="hud-telemetry-item">
                  <span className="hud-telemetry-label">SPEED</span>
                  <span className="hud-telemetry-val">
                    3.2<em>kn</em>
                  </span>
                </div>
                <div className="hud-telemetry-item">
                  <span className="hud-telemetry-label">SONAR</span>
                  <span className="hud-telemetry-val">
                    900<em>kHz</em>
                  </span>
                </div>
                <div className="hud-telemetry-item">
                  <span className="hud-telemetry-label">STATUS</span>
                  <span className="hud-telemetry-val" style={{ color: 'var(--cin-teal)' }}>
                    ACTIVE
                  </span>
                </div>
              </div>
            </div>

            {/* Target Detection Bounding Box Marker (Cinematic UI Demo) */}
            {targetDetected && (
              <div className="hud-target-marker-box">
                <span className="hud-target-marker-label">MILCO 86% · TARGET SX-014</span>
              </div>
            )}

            {/* Target Detection HUD Card */}
            {targetDetected && (
              <div className="hud-target-detect-card">
                <div className="target-card-header">
                  <div className="target-card-badge">
                    <span className="target-card-dot" />
                    <span>ACOUSTIC RETURN DETECTED</span>
                  </div>
                  <span className="target-id-code">TARGET SX-014</span>
                </div>

                <div className="target-card-class-name">MILCO</div>
                <div className="target-card-taxonomy">Mine-Like Contact · Acoustic Shadow Verified</div>

                <div className="target-card-grid">
                  <div className="target-card-metric">
                    <span>CLASS</span>
                    <b>MILCO</b>
                  </div>
                  <div className="target-card-metric">
                    <span>CONFIDENCE</span>
                    <b>86%</b>
                  </div>
                  <div className="target-card-metric">
                    <span>RISK</span>
                    <b className="risk-amber">MEDIUM</b>
                  </div>
                  <div className="target-card-metric">
                    <span>LOCATION</span>
                    <b>SIMULATED</b>
                  </div>
                </div>

                <div className="target-honesty-disclaimer">
                  AI ILLUSTRATIVE VISUALIZATION · Simulated target lock demonstration. Real model inference accessible in mission console.
                </div>
              </div>
            )}

            {/* Bottom HUD Bar */}
            <div className="hud-bottom-bar">
              <div className="hud-status-cluster">
                <div className="hud-status-item">
                  <span className="hud-status-dot" />
                  <span className="hud-status-label">SONAR ARRAY</span>
                  <span className="hud-status-val">ONLINE</span>
                </div>
                <div className="hud-status-item">
                  <span className="hud-status-dot" />
                  <span className="hud-status-label">AI INFERENCE</span>
                  <span className="hud-status-val">ACTIVE</span>
                </div>
                <div className="hud-status-item">
                  <span className="hud-status-dot" />
                  <span className="hud-status-label">TACTICAL LINK</span>
                  <span className="hud-status-val">ESTABLISHED</span>
                </div>
              </div>

              <div style={{ display: 'flex', gap: '10px' }}>
                <button
                  className="btn-hero-primary"
                  style={{ padding: '8px 18px', fontSize: '0.75rem' }}
                  onClick={onEnterMission}
                >
                  ENTER APPLICATION <ArrowRight size={13} />
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Hero Branding & Content Over Video */}
        {bootComplete && (
          <div className="cinematic-hero-content">
            <div className="hero-kicker-tag">
              <Radio size={13} className="pulse-beacon" /> ACOUSTIC INTELLIGENCE PLATFORM
            </div>

            <h1 className="hero-main-title">SONARIS-X</h1>

            <h2 className="hero-secondary-title">
              AI-POWERED UNDERWATER
              <br />
              <span>ACOUSTIC INTELLIGENCE</span>
            </h2>

            <p className="hero-tagline-motto">
              DETECT <span>•</span> IDENTIFY <span>•</span> LOCALIZE <span>•</span> EXPLAIN
            </p>

            <p className="hero-lead-description">
              Turning side-scan sonar imagery into reliable underwater target intelligence.
            </p>

            <div className="hero-cta-group">
              <button className="btn-hero-primary" onClick={onEnterMission}>
                ENTER MISSION <ArrowRight size={15} />
              </button>

              <button className="btn-hero-secondary" onClick={handleScrollToExplore}>
                EXPLORE SYSTEM <ArrowDown size={15} />
              </button>
            </div>
          </div>
        )}

        {/* Scroll Indicator */}
        {bootComplete && (
          <div className="hero-scroll-indicator" onClick={handleScrollToExplore}>
            <span>SCROLL TO EXPLORE MISSION</span>
            <ArrowDown size={14} />
          </div>
        )}
      </section>

      {/* ===================================================================
          3. SCROLL STORY SECTIONS (SECTIONS 01 – 04 + CTA)
          =================================================================== */}
      <LandingScrollSections
        onEnterMission={onEnterMission}
        onExploreDemo={(scenario) => {
          if (onExploreDemo) onExploreDemo(scenario);
          onEnterMission();
        }}
      />
    </div>
  );
};
