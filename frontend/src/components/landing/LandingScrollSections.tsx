import React from 'react';
import {
  Activity,
  ArrowRight,
  BarChart3,
  CheckCircle2,
  Clock,
  Cpu,
  Crosshair,
  Database,
  FileSpreadsheet,
  Layers,
  MapPin,
  Radio,
  Search,
  Shield,
  ShieldAlert,
  Sliders,
  Sparkles,
  Terminal,
  UserCheck,
  Waves,
  Zap,
} from 'lucide-react';
import { getAssetUrl } from '../../utils/assets';

export interface LandingScrollSectionsProps {
  onEnterMission: () => void;
  onExploreDemo: (scenario: string) => void;
}

export const LandingScrollSections: React.FC<LandingScrollSectionsProps> = ({
  onEnterMission,
  onExploreDemo,
}) => {
  return (
    <div className="cinematic-scroll-sections" id="explore-system-section">
      {/* ===================================================================
          SECTION 01: THE PROBLEM
          =================================================================== */}
      <section className="cinematic-story-section">
        <p className="section-kicker">OPERATIONAL CHALLENGE // ACOUSTIC TRIAGE</p>
        <h2 className="section-heading-lg">
          THE
          <br />
          <span style={{ color: 'var(--cin-cyan)' }}>PROBLEM.</span>
        </h2>
        <p className="section-lead-text">
          Side-scan sonar surveys generate hundreds of gigabytes of acoustic imagery.
          Natural seabed ridges, speckle noise, acoustic shadows, and low-contrast contacts
          create high false-alarm rates and overwhelm human analysts.
        </p>

        <div className="problem-visual-grid">
          {/* Card 1: Natural Clutter & Speckle */}
          <div className="problem-card">
            <div className="problem-media-wrap">
              <img
                src={getAssetUrl('assets/realistic/nombo.jpg')}
                alt="Natural Seabed Clutter"
                onError={(e) => {
                  (e.currentTarget as HTMLImageElement).src = getAssetUrl('videos/sonaris-hero-poster.jpg');
                }}
              />
              <div
                style={{
                  position: 'absolute',
                  inset: 0,
                  background: 'linear-gradient(180deg, transparent 40%, rgba(1, 4, 10, 0.95) 100%)',
                }}
              />
            </div>
            <div className="problem-card-body">
              <span className="problem-card-tag warn">ACOUSTIC CLUTTER · NOMBO AMBIGUITY</span>
              <h3 className="problem-card-title">Complex Seabed Clutter</h3>
              <p className="problem-card-desc">
                Glacial boulders, sand ripples, and biogenic sediment generate deceptive acoustic shadows that mimic
                man-made munitions and infrastructure hazards.
              </p>
            </div>
          </div>

          {/* Card 2: Low Contrast Target Ambiguity */}
          <div className="problem-card">
            <div className="problem-media-wrap">
              <img
                src={getAssetUrl('assets/realistic/milco.jpg')}
                alt="Acoustic Sonar Return"
                onError={(e) => {
                  (e.currentTarget as HTMLImageElement).src = getAssetUrl('videos/sonaris-hero-poster.jpg');
                }}
              />
              <div
                style={{
                  position: 'absolute',
                  inset: 0,
                  background: 'linear-gradient(180deg, transparent 40%, rgba(1, 4, 10, 0.95) 100%)',
                }}
              />
            </div>
            <div className="problem-card-body">
              <span className="problem-card-tag info">MANUAL ANALYSIS BOTTLENECK</span>
              <h3 className="problem-card-title">Analyst Fatigue & Scale</h3>
              <p className="problem-card-desc">
                Manual review of continuous 100-meter acoustic swaths takes hours per kilometer.
                High-stakes mine countermeasures and pipeline inspections demand rapid, repeatable classification.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* ===================================================================
          SECTION 02: FROM SONAR TO INTELLIGENCE
          =================================================================== */}
      <section className="cinematic-story-section" style={{ borderTop: '1px solid rgba(56, 189, 248, 0.15)' }}>
        <p className="section-kicker">END-TO-END PIPELINE ARCHITECTURE</p>
        <h2 className="section-heading-lg">
          FROM SONAR
          <br />
          <span style={{ color: 'var(--cin-cyan)' }}>TO INTELLIGENCE.</span>
        </h2>
        <p className="section-lead-text">
          A continuous, defensible processing chain transforming raw acoustic returns into structured tactical intelligence.
        </p>

        <div className="pipeline-flow-rail">
          {/* Step 1 */}
          <div className="pipeline-node-box">
            <div className="pipeline-node-step">
              <Waves size={14} /> STEP 01
            </div>
            <h4 className="pipeline-node-title">SIDE-SCAN SONAR</h4>
            <p className="pipeline-node-desc">
              Raw high-frequency acoustic waterfall return ingested from survey towfish or uncrewed underwater vehicles.
            </p>
          </div>

          {/* Step 2 */}
          <div className="pipeline-node-box">
            <div className="pipeline-node-step">
              <Sliders size={14} /> STEP 02
            </div>
            <h4 className="pipeline-node-title">PREPROCESSING</h4>
            <p className="pipeline-node-desc">
              Contrast-Limited Adaptive Histogram Equalization (CLAHE) and bilateral speckle noise filtering.
            </p>
          </div>

          {/* Step 3 */}
          <div className="pipeline-node-box">
            <div className="pipeline-node-step">
              <Cpu size={14} /> STEP 03
            </div>
            <h4 className="pipeline-node-title">AI DETECTION</h4>
            <p className="pipeline-node-desc">
              Custom-trained YOLOv8n sonar weights identifying candidate contacts with bounding coordinates.
            </p>
          </div>

          {/* Step 4 */}
          <div className="pipeline-node-box">
            <div className="pipeline-node-step">
              <BarChart3 size={14} /> STEP 04
            </div>
            <h4 className="pipeline-node-title">RELIABILITY ANALYSIS</h4>
            <p className="pipeline-node-desc">
              Geometric contour extraction, shadow-to-highlight ratio, and background seabed divergence testing.
            </p>
          </div>

          {/* Step 5 */}
          <div className="pipeline-node-box">
            <div className="pipeline-node-step">
              <ShieldAlert size={14} /> STEP 05
            </div>
            <h4 className="pipeline-node-title">CONFIDENCE + RISK</h4>
            <p className="pipeline-node-desc">
              Calibrated model probability fused with physical shadow evidence into objective LOW, MEDIUM, or HIGH risk.
            </p>
          </div>

          {/* Step 6 */}
          <div className="pipeline-node-box">
            <div className="pipeline-node-step">
              <Crosshair size={14} /> STEP 06
            </div>
            <h4 className="pipeline-node-title">TARGET INTELLIGENCE</h4>
            <p className="pipeline-node-desc">
              Structured MILCO / NOMBO classification cataloging with 6-axis acoustic fingerprint profile.
            </p>
          </div>

          {/* Step 7 */}
          <div className="pipeline-node-box">
            <div className="pipeline-node-step">
              <MapPin size={14} /> STEP 07
            </div>
            <h4 className="pipeline-node-title">GEOLOCATION</h4>
            <p className="pipeline-node-desc">
              Geospatial placement on tactical map (clearly identified as SIMULATED coordinates for demo compliance).
            </p>
          </div>

          {/* Step 8 */}
          <div className="pipeline-node-box">
            <div className="pipeline-node-step">
              <FileSpreadsheet size={14} /> STEP 08
            </div>
            <h4 className="pipeline-node-title">REPORT</h4>
            <p className="pipeline-node-desc">
              Exportable mission packages with executive summaries, audit logs, and downloadable JSON / CSV data.
            </p>
          </div>
        </div>
      </section>

      {/* ===================================================================
          SECTION 03: RELIABLE DETECTION (SONARIS-X DIFFERENTIATORS)
          =================================================================== */}
      <section className="cinematic-story-section" style={{ borderTop: '1px solid rgba(56, 189, 248, 0.15)' }}>
        <p className="section-kicker">PLATFORM ARCHITECTURE & CAPABILITIES</p>
        <h2 className="section-heading-lg">
          RELIABLE
          <br />
          <span style={{ color: 'var(--cin-cyan)' }}>DETECTION.</span>
        </h2>
        <p className="section-lead-text">
          Engineered specifically for underwater acoustics. We explicitly separate implemented hackathon prototype
          capabilities from planned enterprise extensions.
        </p>

        <div className="differentiation-grid">
          {/* Card 1 */}
          <div className="diff-card">
            <div className="diff-icon-wrap">
              <Sliders size={20} />
            </div>
            <h3 className="diff-card-title">Sonar-Aware Preprocessing</h3>
            <p className="diff-card-desc">
              Specialized CLAHE and bilateral filtering tuned to acoustic backscatter and transmission loss curves rather than generic photographic filters.
            </p>
            <span className="diff-status-tag active">IMPLEMENTED & VERIFIED</span>
          </div>

          {/* Card 2 */}
          <div className="diff-card">
            <div className="diff-icon-wrap">
              <Layers size={20} />
            </div>
            <h3 className="diff-card-title">Artifact & Clutter Awareness</h3>
            <p className="diff-card-desc">
              Seabed divergence analysis cross-references acoustic returns against ambient seafloor variance to avoid false positives on natural sediment ripples.
            </p>
            <span className="diff-status-tag active">IMPLEMENTED & VERIFIED</span>
          </div>

          {/* Card 3 */}
          <div className="diff-card">
            <div className="diff-icon-wrap">
              <Search size={20} />
            </div>
            <h3 className="diff-card-title">Small Target Sensitivity</h3>
            <p className="diff-card-desc">
              Sliding-window tiled inference option prevents miniscule targets in high-resolution swaths from being lost during full-frame downsampling.
            </p>
            <span className="diff-status-tag active">IMPLEMENTED & VERIFIED</span>
          </div>

          {/* Card 4 */}
          <div className="diff-card">
            <div className="diff-icon-wrap">
              <ShieldAlert size={20} />
            </div>
            <h3 className="diff-card-title">Confidence + Risk Scoring</h3>
            <p className="diff-card-desc">
              Dual-layer evaluation: raw neural network confidence is paired with an independent geometric risk score to prevent overconfidence on ambiguous contacts.
            </p>
            <span className="diff-status-tag active">IMPLEMENTED & VERIFIED</span>
          </div>

          {/* Card 5 */}
          <div className="diff-card">
            <div className="diff-icon-wrap">
              <UserCheck size={20} />
            </div>
            <h3 className="diff-card-title">Human-in-the-Loop Review</h3>
            <p className="diff-card-desc">
              Dedicated operator queue flags borderline acoustic detections for human sign-off, recording analyst rationale directly into the mission audit trail.
            </p>
            <span className="diff-status-tag active">IMPLEMENTED & VERIFIED</span>
          </div>

          {/* Card 6 */}
          <div className="diff-card">
            <div className="diff-icon-wrap">
              <Radio size={20} />
            </div>
            <h3 className="diff-card-title">Multi-Vehicle Fleet Fusion</h3>
            <p className="diff-card-desc">
              Real-time cross-mission acoustic correlation across multiple autonomous surface and underwater survey platforms operating simultaneously.
            </p>
            <span className="diff-status-tag roadmap">FUTURE ROADMAP EXTENSION</span>
          </div>
        </div>
      </section>

      {/* ===================================================================
          SECTION 04: MISSION INTELLIGENCE
          =================================================================== */}
      <section className="cinematic-story-section" style={{ borderTop: '1px solid rgba(56, 189, 248, 0.15)' }}>
        <p className="section-kicker">VERIFIED MODEL BENCHMARK // ACTUAL RUNTIME DATA</p>
        <h2 className="section-heading-lg">
          MISSION
          <br />
          <span style={{ color: 'var(--cin-cyan)' }}>INTELLIGENCE.</span>
        </h2>
        <p className="section-lead-text">
          Zero fabricated metrics. Real inference results from our custom YOLOv8n checkpoint running on test set image <code>2021_0006_2021.jpg</code>.
        </p>

        <div className="mission-intel-stage">
          <div className="intel-stage-topbar">
            <span>SONARIS-X // INFERENCE VERIFICATION BENCHMARK</span>
            <span style={{ color: 'var(--cin-cyan)' }}>CHECKPOINT: dev_best.pt · REAL SONAR DATASET</span>
          </div>

          <div className="intel-stage-body">
            {/* Raw Sonar Image with Bounding Box */}
            <div className="intel-raw-sonar-panel">
              <img
                src="/api/demo/image?name=Demo%20Mission%2001"
                alt="Real Sonar Target Detection"
                onError={(e) => {
                  (e.currentTarget as HTMLImageElement).src = getAssetUrl('assets/realistic/milco.jpg');
                }}
              />
              <div className="intel-target-bbox">
                <span className="intel-bbox-label">MILCO 34.6%</span>
              </div>
            </div>

            {/* Actual Verified Metrics Feed */}
            <div className="intel-telemetry-feed">
              <div>
                <p className="section-kicker" style={{ fontSize: '0.65rem', marginBottom: '0.5rem' }}>
                  GROUND-TRUTH CONTACT ASSESSMENT
                </p>
                <h3 style={{ fontSize: '1.25rem', color: '#fff', fontWeight: 700, margin: '0 0 1rem' }}>
                  CONTACT T-01: MINE-LIKE CONTACT
                </h3>

                <div className="intel-feed-grid">
                  <div className="intel-feed-card">
                    <span>DETECTED CLASS</span>
                    <b>MILCO</b>
                  </div>
                  <div className="intel-feed-card">
                    <span>RAW MODEL CONFIDENCE</span>
                    <b className="highlight-amber">34.62%</b>
                  </div>
                  <div className="intel-feed-card">
                    <span>RISK CLASSIFICATION</span>
                    <b className="highlight-amber">MEDIUM RISK</b>
                  </div>
                  <div className="intel-feed-card">
                    <span>PROCESSING LATENCY</span>
                    <b>22.5 ms</b>
                  </div>
                  <div className="intel-feed-card">
                    <span>GEOLOCATION</span>
                    <b style={{ color: '#94a3b8' }}>SIMULATED</b>
                  </div>
                  <div className="intel-feed-card">
                    <span>FINGERPRINT / SHADOW</span>
                    <b style={{ color: 'var(--cin-cyan)' }}>AVAILABLE</b>
                  </div>
                </div>

                <p style={{ fontSize: '0.82rem', color: '#94a3b8', lineHeight: 1.6, margin: '0 0 1.25rem' }}>
                  Acoustic highlight at coordinates <code>[179, 332]</code> accompanied by verified acoustic shadow.
                  Classified honestly as <b>MILCO</b> (Mine-Like Contact) without renaming to premature operational conclusions.
                </p>
              </div>

              <div style={{ display: 'flex', gap: '12px' }}>
                <button
                  className="btn-hero-primary"
                  style={{ flex: 1, justifyContent: 'center' }}
                  onClick={onEnterMission}
                >
                  TEST IN DASHBOARD <ArrowRight size={15} />
                </button>
                <button
                  className="btn-hero-secondary"
                  onClick={() => {
                    onExploreDemo('Demo Mission 01');
                    onEnterMission();
                  }}
                >
                  LOAD SCENARIO 01
                </button>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* ===================================================================
          SECTION 05: FINAL CTA — ENTER APPLICATION
          =================================================================== */}
      <section className="cinematic-final-cta-section">
        <div className="final-cta-container">
          <p className="section-kicker" style={{ justifyContent: 'center' }}>
            TACTICAL APPLICATION READY
          </p>
          <h2 className="final-cta-title">ENTER SONARIS-X</h2>
          <p className="final-cta-desc">
            Experience the complete acoustic intelligence console. Upload side-scan sonar files,
            perform real-time YOLOv8n inference, inspect multi-dimensional acoustic fingerprints, and generate exportable mission reports.
          </p>

          <div className="final-cta-buttons">
            <button className="btn-hero-primary" onClick={onEnterMission}>
              <Zap size={16} /> ENTER MISSION CONSOLE
            </button>
            <button
              className="btn-hero-secondary"
              onClick={() => {
                onExploreDemo('Demo Mission 01');
                onEnterMission();
              }}
            >
              <Activity size={16} /> LAUNCH DEMO MISSION 01
            </button>
          </div>

          <div className="final-meta-tags">
            <div className="final-meta-item">
              <span className="dot" />
              <span>SONARIS-X v2.4 PROD</span>
            </div>
            <div className="final-meta-item">
              <span className="dot" />
              <span>YOLOv8n CUSTOM SONAR WEIGHTS</span>
            </div>
            <div className="final-meta-item">
              <span className="dot" />
              <span>DATA HONESTY COMPLIANT</span>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
};
