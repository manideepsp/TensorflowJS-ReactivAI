import { describe, expect, it } from 'vitest';
import { EngagementEngine } from './engagementEngine';

describe('EngagementEngine', () => {
  it('computes weighted score and components', () => {
    const engine = new EngagementEngine();
    const score = engine.computeScore({
      emotionConfidence: 0.5,
      normalizedVoiceEnergy: 0.5,
      speechContinuity: 0.5,
    });

    // 0.5*(0.4+0.3+0.3)=0.5 => 50
    expect(score.overall).toBe(50);
    expect(score.emotionComponent).toBe(20);
    expect(score.voiceComponent).toBe(15);
    expect(score.speechComponent).toBe(15);
  });

  it('clamps inputs to valid range', () => {
    const engine = new EngagementEngine();
    const score = engine.computeScore({
      emotionConfidence: 2,
      normalizedVoiceEnergy: -1,
      speechContinuity: 0.4,
    });

    // emotion -> 1, voice -> 0
    expect(score.overall).toBe(52);
  });

  it('tracks speech continuity over rolling buffer', () => {
    const engine = new EngagementEngine();
    const pattern = [true, false, true, true, false];
    pattern.forEach(p => engine.updateSpeechContinuity(p));

    expect(engine.getSpeechContinuity()).toBeCloseTo(3 / 5, 5);
  });
});
