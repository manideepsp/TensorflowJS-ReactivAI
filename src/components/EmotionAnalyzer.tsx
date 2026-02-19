/**
 * EmotionAnalyzer Component
 * Main React component for real-time emotion and engagement analysis
 * STRICT CLIENT-SIDE ONLY - No SSR
 */

import { useEffect, useRef, useState } from 'react';
import * as tf from '@tensorflow/tfjs';
import { initializeTensorFlow } from '../ml/tfSetup';
import {
  initializeFaceDetector,
  detectFace,
  type FaceDetectionResult,
} from '../ml/faceDetector';
import {
  loadEmotionModel,
  predictEmotion,
  getTopEmotion,
  EMOTION_LABELS,
} from '../ml/emotionClassifier';
import { TemporalSmoother } from '../ml/temporalSmoother';
import { AudioEngine, type AudioMetrics } from '../audio/audioEngine';
import {
  EngagementEngine,
  type EngagementScore,
} from '../scoring/engagementEngine';
import {
  PerformanceMonitor,
  type PerformanceMetrics,
} from '../monitoring/performanceMonitor';

interface EmotionPrediction {
  label: string;
  confidence: number;
  probabilities: Float32Array;
}

type InitStage =
  | 'tensorflow'
  | 'face-detector'
  | 'emotion-model'
  | 'audio'
  | 'camera'
  | 'done';

export default function EmotionAnalyzer() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationFrameRef = useRef<number | null>(null);

  const [initStage, setInitStage] = useState<InitStage>('tensorflow');
  const [isInitializing, setIsInitializing] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [isRunning, setIsRunning] = useState(false);

  const [emotion, setEmotion] = useState<EmotionPrediction | null>(null);
  const [engagement, setEngagement] = useState<EngagementScore | null>(null);
  const [audioMetrics, setAudioMetrics] = useState<AudioMetrics | null>(null);
  const [perfMetrics, setPerfMetrics] = useState<PerformanceMetrics | null>(null);

  // Engine instances
  const audioEngineRef = useRef<AudioEngine | null>(null);
  const engagementEngineRef = useRef<EngagementEngine | null>(null);
  const performanceMonitorRef = useRef<PerformanceMonitor | null>(null);
  const temporalSmootherRef = useRef<TemporalSmoother | null>(null);

  /**
   * Initialize all engines
   * The video element is ALWAYS in the DOM (see render below),
   * so videoRef.current is guaranteed to be available.
   */
  useEffect(() => {
    let mounted = true;

    async function initialize() {
      try {
        console.log('Initializing EdgePresence...');

        // 1. TensorFlow.js
        setInitStage('tensorflow');
        await initializeTensorFlow();
        if (!mounted) return;
        console.log('✓ TensorFlow.js initialized');

        // 2. Face detector
        setInitStage('face-detector');
        await initializeFaceDetector();
        if (!mounted) return;
        console.log('✓ Face detector initialized');

        // 3. Emotion model
        setInitStage('emotion-model');
        await loadEmotionModel();
        if (!mounted) return;
        console.log('✓ Emotion model loaded');

        // 4. Audio engine
        setInitStage('audio');
        audioEngineRef.current = new AudioEngine();
        await audioEngineRef.current.initialize();
        if (!mounted) return;
        console.log('✓ Audio engine initialized');

        // 5. Scoring & monitoring engines (synchronous)
        engagementEngineRef.current = new EngagementEngine();
        performanceMonitorRef.current = new PerformanceMonitor();
        temporalSmootherRef.current = new TemporalSmoother(0.3);

        // 6. Camera – videoRef is always in the DOM now
        setInitStage('camera');
        const video = videoRef.current;
        if (!video) {
          // Extremely unlikely since element is always rendered, but guard anyway
          throw new Error('Video element not found after mount');
        }

        const stream = await navigator.mediaDevices.getUserMedia({
          video: {
            width: { ideal: 640 },
            height: { ideal: 480 },
            facingMode: 'user',
          },
        });
        if (!mounted) {
          stream.getTracks().forEach((t) => t.stop());
          return;
        }

        video.srcObject = stream;
        // Wait for metadata so videoWidth/videoHeight are available
        await new Promise<void>((resolve) => {
          if (video.readyState >= HTMLMediaElement.HAVE_METADATA) {
            resolve();
          } else {
            video.addEventListener('loadedmetadata', () => resolve(), { once: true });
          }
        });
        // Play the video (muted + playsInline so autoplay is allowed)
        await video.play();
        if (!mounted) return;

        console.log('✓ Camera stream active');

        setInitStage('done');
        setIsInitializing(false);
        setIsRunning(true);
        console.log('✓ Initialization complete');
      } catch (err) {
        console.error('Initialization error:', err);
        if (mounted) {
          setError(err instanceof Error ? err.message : 'Failed to initialize');
          setIsInitializing(false);
        }
      }
    }

    initialize();

    return () => {
      mounted = false;
    };
  }, []);

  /**
   * Main processing loop
   */
  useEffect(() => {
    if (!isRunning) return;

    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Set canvas size to match video
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;

    let running = true;
    let frameCount = 0;

    async function processFrame() {
      if (!running || !video || !canvas) return;

      const perfMonitor = performanceMonitorRef.current;
      const audioEngine = audioEngineRef.current;
      const engagementEngine = engagementEngineRef.current;
      const smoother = temporalSmootherRef.current;

      // If any engine is missing, skip this frame but keep the loop alive
      if (!perfMonitor || !audioEngine || !engagementEngine || !smoother) {
        if (running) {
          animationFrameRef.current = requestAnimationFrame(processFrame);
        }
        return;
      }

      try {
        frameCount++;

        // Update FPS
        perfMonitor.updateFps();

        // Detect face
        const faceStart = performance.now();
        const face = await detectFace(video);
        perfMonitor.recordFaceDetectionLatency(performance.now() - faceStart);

        if (face) {
          // Extract face region (sync, wrapped in tidy to auto-dispose intermediates)
          const faceImage = extractFaceRegion(video, face);

          // Predict emotion (async — we dispose faceImage manually)
          const emotionStart = performance.now();
          const rawProbs = await predictEmotion(faceImage);
          faceImage.dispose();

          // Apply temporal smoothing (pure JS, no tensors)
          const smoothedProbs = smoother.smooth(rawProbs);
          const top = getTopEmotion(smoothedProbs);

          perfMonitor.recordEmotionLatency(performance.now() - emotionStart);

          setEmotion({
            label: top.label,
            confidence: top.confidence,
            probabilities: smoothedProbs,
          });

          // Process audio (pure JS, no tensors)
          const audioStart = performance.now();
          const audio = audioEngine.getMetrics();
          perfMonitor.recordAudioLatency(performance.now() - audioStart);
          setAudioMetrics(audio);

          // Update speech continuity
          const speechContinuity = engagementEngine.updateSpeechContinuity(
            audio.isSpeaking,
          );

          // Compute engagement score
          const engagementScore = engagementEngine.computeScore({
            emotionConfidence: top.confidence,
            normalizedVoiceEnergy: audio.normalizedEnergy,
            speechContinuity,
          });
          setEngagement(engagementScore);

          // Draw face mesh overlay
          drawFaceMesh(ctx, face, canvas.width, canvas.height);
        } else {
          // No face detected – clear overlay
          ctx.clearRect(0, 0, canvas.width, canvas.height);
          setEmotion(null);
        }

        // Update performance metrics
        const metrics = perfMonitor.getMetrics();
        setPerfMetrics(metrics);
      } catch (err) {
        console.error('Frame processing error:', err);
      }

      // Schedule next frame
      if (running) {
        animationFrameRef.current = requestAnimationFrame(processFrame);
      }
    }

    // Start processing loop
    processFrame();

    return () => {
      running = false;
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [isRunning]);

  /**
   * Cleanup on unmount
   */
  useEffect(() => {
    return () => {
      // Stop video stream
      if (videoRef.current?.srcObject) {
        const stream = videoRef.current.srcObject as MediaStream;
        stream.getTracks().forEach((track) => track.stop());
      }

      // Dispose audio engine
      audioEngineRef.current?.dispose();

      // NOTE: Do NOT call tf.dispose() here!
      // It destroys ALL tensors globally, including the face detector's
      // internal model weights.  The module-level `detector` reference in
      // faceDetector.ts survives across React remounts / HMR, pointing at a
      // zombie detector whose tensors are gone → estimateFaces() returns 0.
      // initializeFaceDetector() now handles its own disposal + re-creation.
    };
  }, []);

  /**
   * Extract face region and preprocess for emotion model
   */
  function extractFaceRegion(
    video: HTMLVideoElement,
    face: FaceDetectionResult,
  ): tf.Tensor4D {
    return tf.tidy(() => {
      // Create temporary canvas for face extraction
      const tempCanvas = document.createElement('canvas');
      tempCanvas.width = 48;
      tempCanvas.height = 48;
      const tempCtx = tempCanvas.getContext('2d')!;

      // Draw face region
      const { xMin, yMin, width, height } = face.box;
      tempCtx.drawImage(video, xMin, yMin, width, height, 0, 0, 48, 48);

      // Get image data and convert to grayscale
      const imageData = tempCtx.getImageData(0, 0, 48, 48);
      const grayscale = new Float32Array(48 * 48);

      for (let i = 0; i < 48 * 48; i++) {
        const idx = i * 4;
        const gray =
          0.299 * imageData.data[idx] +
          0.587 * imageData.data[idx + 1] +
          0.114 * imageData.data[idx + 2];
        grayscale[i] = gray / 255; // Normalize to 0-1
      }

      // Create tensor [1, 48, 48, 1]
      return tf.tensor4d(grayscale, [1, 48, 48, 1]);
    });
  }

  /**
   * Draw face mesh overlay
   */
  function drawFaceMesh(
    ctx: CanvasRenderingContext2D,
    face: FaceDetectionResult,
    width: number,
    height: number,
  ): void {
    // Clear canvas
    ctx.clearRect(0, 0, width, height);

    // Draw bounding box
    const { xMin, yMin, width: boxWidth, height: boxHeight } = face.box;
    ctx.strokeStyle = '#5df2d6';
    ctx.lineWidth = 2;
    ctx.strokeRect(xMin, yMin, boxWidth, boxHeight);

    // Draw keypoints
    ctx.fillStyle = '#5df2d6';
    face.keypoints.forEach((kp) => {
      ctx.beginPath();
      ctx.arc(kp.x, kp.y, 1, 0, 2 * Math.PI);
      ctx.fill();
    });
  }

  /** Human-readable init stage labels */
  const stageLabels: Record<InitStage, string> = {
    tensorflow: 'Loading TensorFlow.js…',
    'face-detector': 'Loading face detector…',
    'emotion-model': 'Loading emotion model…',
    audio: 'Requesting microphone…',
    camera: 'Requesting camera…',
    done: 'Ready',
  };

  // ─── Render ──────────────────────────────────────────────────────────
  // The video + canvas elements are ALWAYS in the DOM so that refs are
  // available during initialisation.  The loading overlay sits on top.

  return (
    <div
      style={{
        display: 'grid',
        gridTemplateColumns: isInitializing || error ? '1fr' : '1fr 1fr',
        gap: '24px',
        alignItems: 'start',
        color: 'var(--text)',
      }}
    >
      {/* Video Feed – always rendered so videoRef is never null */}
      <div
        style={{
          position: 'relative',
          borderRadius: '16px',
          overflow: 'hidden',
          border: '1px solid rgba(255,255,255,0.08)',
        }}
      >
        <video
          ref={videoRef}
          style={{
            width: '100%',
            maxWidth: '680px',
            background: '#05060f',
            display: 'block',
          }}
          muted
          playsInline
        />
        <canvas
          ref={canvasRef}
          style={{
            position: 'absolute',
            inset: 0,
            width: '100%',
            maxWidth: '680px',
            pointerEvents: 'none',
          }}
        />

        {/* Loading / error overlay on top of video */}
        {isInitializing && (
          <div
            style={{
              position: 'absolute',
              inset: 0,
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              background: 'rgba(5, 6, 15, 0.85)',
              backdropFilter: 'blur(4px)',
              zIndex: 10,
            }}
          >
            <h2 style={{ margin: '0 0 8px 0' }}>Initializing EdgePresence</h2>
            <p style={{ color: 'var(--muted)', margin: 0 }}>
              {stageLabels[initStage]}
            </p>
          </div>
        )}

        {error && (
          <div
            style={{
              position: 'absolute',
              inset: 0,
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              background: 'rgba(5, 6, 15, 0.9)',
              zIndex: 10,
              padding: '20px',
            }}
          >
            <h2 style={{ color: '#ff6b6b', margin: '0 0 8px 0' }}>Error</h2>
            <p style={{ color: 'var(--muted)', margin: 0, textAlign: 'center' }}>
              {error}
            </p>
          </div>
        )}
      </div>

      {/* Metrics Panel – shown only when running */}
      {!isInitializing && !error && (
        <div style={{ display: 'grid', gap: '16px' }}>
          {/* Emotion Display */}
          {emotion ? (
            <div
              style={{
                padding: '16px',
                borderRadius: '14px',
                background: 'var(--panel)',
                border: '1px solid rgba(255,255,255,0.12)',
                boxShadow: '0 10px 30px rgba(0,0,0,0.25)',
              }}
            >
              <h2
                style={{
                  margin: '0 0 10px 0',
                  letterSpacing: '-0.01em',
                }}
              >
                Emotion: {emotion.label}
              </h2>
              <p
                style={{
                  fontSize: '24px',
                  margin: '10px 0',
                  color: 'var(--accent)',
                }}
              >
                Confidence: {(emotion.confidence * 100).toFixed(1)}%
              </p>

              {/* Probability Bars */}
              <div style={{ marginTop: '15px', display: 'grid', gap: '10px' }}>
                {EMOTION_LABELS.map((label, idx) => (
                  <div key={label}>
                    <div
                      style={{
                        display: 'flex',
                        justifyContent: 'space-between',
                        fontSize: '12px',
                        color: 'var(--muted)',
                      }}
                    >
                      <span>{label}</span>
                      <span>
                        {(emotion.probabilities[idx] * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div
                      style={{
                        width: '100%',
                        height: '14px',
                        background: 'rgba(255,255,255,0.08)',
                        borderRadius: '999px',
                        overflow: 'hidden',
                      }}
                    >
                      <div
                        style={{
                          width: `${emotion.probabilities[idx] * 100}%`,
                          height: '100%',
                          background:
                            idx === EMOTION_LABELS.indexOf(emotion.label)
                              ? 'var(--accent)'
                              : 'rgba(93,242,214,0.45)',
                          transition: 'width 0.25s ease',
                        }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          ) : (
            <div
              style={{
                padding: '16px',
                borderRadius: '14px',
                background: 'var(--panel)',
                border: '1px solid rgba(255,255,255,0.12)',
                boxShadow: '0 10px 30px rgba(0,0,0,0.25)',
                textAlign: 'center',
                color: 'var(--muted)',
              }}
            >
              <p>No face detected – look at the camera</p>
            </div>
          )}

          {/* Engagement Score */}
          {engagement && (
            <div
              style={{
                padding: '16px',
                borderRadius: '14px',
                background: 'var(--panel)',
                border: '1px solid rgba(255,255,255,0.12)',
                boxShadow: '0 10px 30px rgba(0,0,0,0.25)',
              }}
            >
              <h2 style={{ margin: '0 0 10px 0' }}>Engagement Score</h2>
              <p
                style={{
                  fontSize: '40px',
                  fontWeight: 700,
                  margin: '6px 0 14px 0',
                  color: 'var(--accent-2)',
                }}
              >
                {engagement.overall}/100
              </p>

              {/* Overall bar */}
              <div style={{ marginBottom: '14px' }}>
                <div
                  style={{
                    width: '100%',
                    height: '10px',
                    background: 'rgba(255,255,255,0.08)',
                    borderRadius: '999px',
                    overflow: 'hidden',
                  }}
                >
                  <div
                    style={{
                      width: `${engagement.overall}%`,
                      height: '100%',
                      background: 'var(--accent-2)',
                      transition: 'width 0.3s ease',
                    }}
                  />
                </div>
              </div>

              {/* Component bars */}
              <div style={{ display: 'grid', gap: '10px' }}>
                {([
                  { label: 'Emotion', value: engagement.emotionComponent, color: 'var(--accent)' },
                  { label: 'Voice', value: engagement.voiceComponent, color: '#a78bfa' },
                  { label: 'Speech', value: engagement.speechComponent, color: '#60a5fa' },
                ] as const).map(({ label, value, color }) => (
                  <div key={label}>
                    <div
                      style={{
                        display: 'flex',
                        justifyContent: 'space-between',
                        fontSize: '12px',
                        color: 'var(--muted)',
                        marginBottom: '3px',
                      }}
                    >
                      <span>{label}</span>
                      <span>{value}/100</span>
                    </div>
                    <div
                      style={{
                        width: '100%',
                        height: '8px',
                        background: 'rgba(255,255,255,0.08)',
                        borderRadius: '999px',
                        overflow: 'hidden',
                      }}
                    >
                      <div
                        style={{
                          width: `${value}%`,
                          height: '100%',
                          background: color,
                          transition: 'width 0.3s ease',
                        }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Audio Metrics */}
          {audioMetrics && (
            <div
              style={{
                padding: '16px',
                borderRadius: '14px',
                background: 'var(--panel)',
                border: '1px solid rgba(255,255,255,0.12)',
                boxShadow: '0 10px 30px rgba(0,0,0,0.25)',
              }}
            >
              <h3 style={{ margin: '0 0 12px 0' }}>Audio</h3>

              {/* Energy bar */}
              <div style={{ marginBottom: '10px' }}>
                <div
                  style={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    fontSize: '12px',
                    color: 'var(--muted)',
                    marginBottom: '3px',
                  }}
                >
                  <span>Energy</span>
                  <span>{(audioMetrics.normalizedEnergy * 100).toFixed(1)}%</span>
                </div>
                <div
                  style={{
                    width: '100%',
                    height: '8px',
                    background: 'rgba(255,255,255,0.08)',
                    borderRadius: '999px',
                    overflow: 'hidden',
                  }}
                >
                  <div
                    style={{
                      width: `${(audioMetrics.normalizedEnergy * 100).toFixed(1)}%`,
                      height: '100%',
                      background: '#34d399',
                      transition: 'width 0.25s ease',
                    }}
                  />
                </div>
              </div>

              {/* Speaking indicator */}
              <div
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '8px',
                  fontSize: '13px',
                  color: 'var(--muted)',
                }}
              >
                <div
                  style={{
                    width: '10px',
                    height: '10px',
                    borderRadius: '50%',
                    background: audioMetrics.isSpeaking ? '#34d399' : 'rgba(255,255,255,0.15)',
                    boxShadow: audioMetrics.isSpeaking ? '0 0 8px #34d399' : 'none',
                    transition: 'all 0.25s ease',
                  }}
                />
                <span>{audioMetrics.isSpeaking ? 'Speaking' : 'Silent'}</span>
              </div>
            </div>
          )}

          {/* Performance Metrics */}
          {perfMetrics && (
            <div
              style={{
                padding: '16px',
                borderRadius: '14px',
                background: 'var(--panel)',
                border: '1px solid rgba(255,255,255,0.12)',
                boxShadow: '0 10px 30px rgba(0,0,0,0.25)',
                fontSize: '12px',
              }}
            >
              <h3 style={{ margin: '0 0 10px 0' }}>Performance</h3>
              <p>FPS: {perfMetrics.fps}</p>
              <p>Emotion Latency: {perfMetrics.emotionLatency}ms</p>
              <p>Face Detection: {perfMetrics.faceDetectionLatency}ms</p>
              <p>Tensors: {perfMetrics.numTensors}</p>
              <p>Memory: {(perfMetrics.numBytes / 1024 / 1024).toFixed(2)} MB</p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
