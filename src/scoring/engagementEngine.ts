/**
 * Engagement Scoring Engine
 * Computes engagement score from emotion confidence, voice energy, and speech continuity
 */

export interface EngagementInput {
  emotionConfidence: number;
  normalizedVoiceEnergy: number;
  speechContinuity: number;
}

export interface EngagementScore {
  overall: number;
  emotionComponent: number;
  voiceComponent: number;
  speechComponent: number;
  timestamp: number;
}

export class EngagementEngine {
  private readonly emotionWeight: number = 0.4;
  private readonly voiceWeight: number = 0.3;
  private readonly speechWeight: number = 0.3;

  private speechFrameBuffer: boolean[] = [];
  private readonly bufferSize: number = 30; // ~1 second at 30 FPS

  /**
   * Compute engagement score
   * 
   * Formula:
   * Score = (Emotion Confidence × 0.4) + (Voice Energy × 0.3) + (Speech Continuity × 0.3)
   * 
   * All inputs should be normalized to 0-1 range
   * Output is scaled to 0-100
   */
  computeScore(input: EngagementInput): EngagementScore {
    // Clamp inputs to valid range
    const emotionConfidence = this.clamp(input.emotionConfidence, 0, 1);
    const normalizedVoiceEnergy = this.clamp(input.normalizedVoiceEnergy, 0, 1);
    const speechContinuity = this.clamp(input.speechContinuity, 0, 1);

    // Compute weighted components
    const emotionComponent = emotionConfidence * this.emotionWeight;
    const voiceComponent = normalizedVoiceEnergy * this.voiceWeight;
    const speechComponent = speechContinuity * this.speechWeight;

    // Sum and scale to 0-100
    const overall = (emotionComponent + voiceComponent + speechComponent) * 100;

    return {
      overall: Math.round(overall),
      emotionComponent: Math.round(emotionComponent * 100),
      voiceComponent: Math.round(voiceComponent * 100),
      speechComponent: Math.round(speechComponent * 100),
      timestamp: Date.now(),
    };
  }

  /**
   * Update speech continuity from speaking detection
   * Maintains a rolling window of speech frames
   */
  updateSpeechContinuity(isSpeaking: boolean): number {
    // Add current frame to buffer
    this.speechFrameBuffer.push(isSpeaking);

    // Maintain buffer size
    if (this.speechFrameBuffer.length > this.bufferSize) {
      this.speechFrameBuffer.shift();
    }

    // Compute continuity as ratio of speaking frames
    const speakingCount = this.speechFrameBuffer.filter(x => x).length;
    return speakingCount / this.speechFrameBuffer.length;
  }

  /**
   * Reset speech continuity buffer
   */
  resetSpeechContinuity(): void {
    this.speechFrameBuffer = [];
  }

  /**
   * Get current speech continuity value
   */
  getSpeechContinuity(): number {
    if (this.speechFrameBuffer.length === 0) {
      return 0;
    }
    const speakingCount = this.speechFrameBuffer.filter(x => x).length;
    return speakingCount / this.speechFrameBuffer.length;
  }

  /**
   * Clamp value to range
   */
  private clamp(value: number, min: number, max: number): number {
    return Math.min(Math.max(value, min), max);
  }
}
