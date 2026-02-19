/**
 * Audio Engine
 * Client-side audio processing using Web Audio API
 * Computes RMS energy and speech detection
 */

export interface AudioMetrics {
  rmsEnergy: number;
  normalizedEnergy: number;
  isSpeaking: boolean;
  timestamp: number;
}

export class AudioEngine {
  private audioContext: AudioContext | null = null;
  private analyser: AnalyserNode | null = null;
  private microphone: MediaStreamAudioSourceNode | null = null;
  private stream: MediaStream | null = null;
  private dataArray: Float32Array<ArrayBuffer> | null = null;
  
  private readonly smoothingTimeConstant: number = 0.8;
  private readonly fftSize: number = 2048;
  private readonly speakingThreshold: number = 0.02;
  
  private initialized: boolean = false;
  private active: boolean = false;

  /**
   * Initialize audio engine and request microphone access
   */
  async initialize(): Promise<void> {
    if (this.initialized) {
      return;
    }

    try {
      // Request microphone access
      this.stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      });

      // Create audio context
      this.audioContext = new AudioContext();
      
      // Create analyser node
      this.analyser = this.audioContext.createAnalyser();
      this.analyser.fftSize = this.fftSize;
      this.analyser.smoothingTimeConstant = this.smoothingTimeConstant;

      // Connect microphone to analyser
      this.microphone = this.audioContext.createMediaStreamSource(this.stream);
      this.microphone.connect(this.analyser);

      // Allocate buffer for time-domain samples; use ArrayBuffer to satisfy TS lib generics
      this.dataArray = new Float32Array(this.analyser.frequencyBinCount) as Float32Array<ArrayBuffer>;

      this.initialized = true;
      this.active = true;

      console.log('Audio engine initialized');
    } catch (error) {
      console.error('Failed to initialize audio engine:', error);
      throw new Error('Microphone access denied or unavailable');
    }
  }

  /**
   * Compute current audio metrics
   */
  getMetrics(): AudioMetrics {
    if (!this.initialized || !this.analyser || !this.dataArray) {
      return {
        rmsEnergy: 0,
        normalizedEnergy: 0,
        isSpeaking: false,
        timestamp: Date.now(),
      };
    }

    // Get time domain data
    this.analyser.getFloatTimeDomainData(this.dataArray);

    // Compute RMS (Root Mean Square) energy
    let sum = 0;
    for (let i = 0; i < this.dataArray.length; i++) {
      sum += this.dataArray[i] * this.dataArray[i];
    }
    const rmsEnergy = Math.sqrt(sum / this.dataArray.length);

    // Normalize energy to 0-1 range (typical RMS values are 0-0.5)
    const normalizedEnergy = Math.min(rmsEnergy * 2, 1.0);

    // Detect speech via threshold
    const isSpeaking = rmsEnergy > this.speakingThreshold;

    return {
      rmsEnergy,
      normalizedEnergy,
      isSpeaking,
      timestamp: Date.now(),
    };
  }

  /**
   * Check if audio is active
   */
  isActive(): boolean {
    return this.active;
  }

  /**
   * Check if audio is initialized
   */
  isInitialized(): boolean {
    return this.initialized;
  }

  /**
   * Pause audio processing
   */
  pause(): void {
    if (this.audioContext && this.audioContext.state === 'running') {
      this.audioContext.suspend();
      this.active = false;
    }
  }

  /**
   * Resume audio processing
   */
  resume(): void {
    if (this.audioContext && this.audioContext.state === 'suspended') {
      this.audioContext.resume();
      this.active = true;
    }
  }

  /**
   * Dispose audio engine and release resources
   */
  dispose(): void {
    if (this.microphone) {
      this.microphone.disconnect();
      this.microphone = null;
    }

    if (this.analyser) {
      this.analyser.disconnect();
      this.analyser = null;
    }

    if (this.stream) {
      this.stream.getTracks().forEach(track => track.stop());
      this.stream = null;
    }

    if (this.audioContext) {
      this.audioContext.close();
      this.audioContext = null;
    }

    this.dataArray = null;
    this.initialized = false;
    this.active = false;

    console.log('Audio engine disposed');
  }
}
