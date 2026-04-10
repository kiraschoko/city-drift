import React, { useState, useEffect, useRef, useCallback } from "react";
import { AlertCircle, Loader2, Activity, Play, Square, Video, VideoOff } from "lucide-react";
import { motion, AnimatePresence } from "motion/react";
import * as mpHands from "@mediapipe/hands";
import * as mpCamera from "@mediapipe/camera_utils";

// MediaPipe packages often use UMD/CommonJS which doesn't always interop well with Vite's named exports.
const Hands = (mpHands as any).Hands || (mpHands as any).default?.Hands || mpHands;
const Camera = (mpCamera as any).Camera || (mpCamera as any).default?.Camera || mpCamera;

// --- Constants ---
const MAP_SOURCE = "/Map-Berlin-v3.png";
const ANALYSIS_WINDOW_SIZE = 120; // px
const VIEWPORT_SIZE = 440; // px

interface Point {
  x: number;
  y: number;
}

interface AnalysisData {
  green: number;
  water: number;
  urban: number;
}

// --- Components ---

const Waveform = ({ data, color }: { data: number[], color: string }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    if (data.length < 2) return;

    ctx.beginPath();
    ctx.strokeStyle = color;
    ctx.lineWidth = 1;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";

    const step = w / (data.length - 1);
    
    ctx.moveTo(0, h - (data[0] * h * 0.6 + h * 0.2));
    
    for (let i = 1; i < data.length; i++) {
      const x = i * step;
      const y = h - (data[i] * h * 0.6 + h * 0.2);
      
      const xc = (x + (i - 1) * step) / 2;
      const yc = (y + (h - (data[i-1] * h * 0.6 + h * 0.2))) / 2;
      ctx.quadraticCurveTo((i-1) * step, h - (data[i-1] * h * 0.6 + h * 0.2), xc, yc);
    }

    const lastIdx = data.length - 1;
    ctx.lineTo(lastIdx * step, h - (data[lastIdx] * h * 0.6 + h * 0.2));
    
    ctx.stroke();
    
    // Subtle glow
    ctx.shadowBlur = 3;
    ctx.shadowColor = color;
    ctx.stroke();

  }, [data, color]);

  return (
    <div className="w-16 h-4 opacity-40 group-hover:opacity-60 transition-opacity">
      <canvas ref={canvasRef} width={64} height={16} className="w-full h-full" />
    </div>
  );
};

const MetricRow = ({ label, value, color, history }: { label: string, value: number, color: string, history: number[] }) => {
  return (
    <div className="flex flex-col gap-2 w-full transition-opacity duration-300 group">
      <div className="flex items-center justify-between w-full">
        <div className="flex items-center gap-2">
          <div className="w-1 h-1 rounded-full" style={{ backgroundColor: color }} />
          <span className="text-[9px] font-mono tracking-[0.2em] text-[#C7D0D9] transition-colors uppercase">{label}</span>
        </div>
        
        <div className="flex items-center gap-4">
          <Waveform data={history} color={color} />
          
          <div className="flex items-center gap-3">
            {/* Calibrated Measurement Bar */}
            <div className="w-32 h-[1px] bg-[#1A2320] relative overflow-visible">
              {/* Active Fill */}
              <motion.div 
                className="absolute inset-y-0 left-0"
                style={{ backgroundColor: color }}
                initial={{ width: 0 }}
                animate={{ width: `${value}%` }}
                transition={{ type: "spring", stiffness: 40, damping: 20 }}
              >
                {/* Active Edge Glow */}
                <div 
                  className="absolute right-0 top-1/2 -translate-y-1/2 w-[1px] h-[4px] shadow-[0_0_8px_currentColor]" 
                  style={{ color: color, backgroundColor: 'currentColor' }} 
                />
              </motion.div>

              {/* Indicator Head (Tick) */}
              <motion.div
                className="absolute top-1/2 -translate-y-1/2 w-[1px] h-4 bg-white/40 z-10"
                animate={{ 
                  left: `${value}%`,
                  x: [0, 0.3, -0.3, 0] // Subtle micro-jitter
                }}
                transition={{ 
                  left: { type: "spring", stiffness: 40, damping: 20 },
                  x: { repeat: Infinity, duration: 0.15, ease: "linear" }
                }}
              />
            </div>

            <div className="min-w-[32px] text-right">
              <span className="text-[10px] font-mono text-[#8FA0B2] tabular-nums">
                {value.toFixed(0)}%
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

const SpectralMatrix = ({ colors }: { colors: string[] }) => {
  const [displayColors, setDisplayColors] = useState<string[]>(new Array(80).fill('#1A2320'));
  const scanLineRef = useRef(0);
  const targetColorsRef = useRef<string[]>(colors);

  useEffect(() => {
    targetColorsRef.current = colors;
  }, [colors]);

  useEffect(() => {
    const interval = setInterval(() => {
      const GRID_X = 20;
      const GRID_Y = 4;
      const y = scanLineRef.current;
      
      setDisplayColors(prev => {
        const next = [...prev];
        for (let x = 0; x < GRID_X; x++) {
          const idx = y * GRID_X + x;
          if (targetColorsRef.current[idx]) {
            next[idx] = targetColorsRef.current[idx];
          }
        }
        return next;
      });
      
      scanLineRef.current = (scanLineRef.current + 1) % GRID_Y;
    }, 80);

    return () => clearInterval(interval);
  }, []);

  return (
    <div className="w-40 h-8 grid grid-cols-20 grid-rows-4 flex-shrink-0 overflow-hidden">
      {displayColors.map((color, i) => (
        <div 
          key={i} 
          style={{ backgroundColor: color }} 
          className="w-full h-full transition-colors duration-300"
        />
      ))}
    </div>
  );
};

// --- Audio Engine ---

class MapAudioEngine {
  private ctx: AudioContext | null = null;
  private masterGain: GainNode | null = null;
  private mainFilter: BiquadFilterNode | null = null;
  private reverb: ConvolverNode | null = null;
  private reverbGain: GainNode | null = null;
  private noiseGain: GainNode | null = null;
  
  private drumGainNode: GainNode | null = null;
  private bassGainNode: GainNode | null = null;
  private chordGainNode: GainNode | null = null;
  private ambienceGainNode: GainNode | null = null;
  private delayNode: DelayNode | null = null;
  private delayGainNode: GainNode | null = null;
  
  private isPlaying: boolean = false;
  private step: number = 0;
  private nextStepTime: number = 0;
  private tempo: number = 84; // Lofi tempo
  private schedulerTimer: any = null;

  private params: AnalysisData = { green: 0, water: 0, urban: 0 };
  private gestureX: number = 0.5;
  private gestureY: number = 0.5;
  private targetGestureX: number = 0.5;
  private targetGestureY: number = 0.5;
  private isGestureActive: boolean = false;

  constructor() {}

  async init() {
    if (this.ctx) return;
    this.ctx = new (window.AudioContext || (window as any).webkitAudioContext)();
    
    this.masterGain = this.ctx.createGain();
    this.masterGain.gain.value = 0.6;
    
    this.mainFilter = this.ctx.createBiquadFilter();
    this.mainFilter.type = "lowpass";
    this.mainFilter.frequency.value = 4000;
    this.mainFilter.Q.value = 0.7;

    this.reverb = this.ctx.createConvolver();
    this.reverbGain = this.ctx.createGain();
    this.reverbGain.gain.value = 0;

    this.delayNode = this.ctx.createDelay(1.0);
    this.delayNode.delayTime.value = 0.4;
    this.delayGainNode = this.ctx.createGain();
    this.delayGainNode.gain.value = 0;

    // Layer Gains
    this.drumGainNode = this.ctx.createGain();
    this.bassGainNode = this.ctx.createGain();
    this.chordGainNode = this.ctx.createGain();
    this.ambienceGainNode = this.ctx.createGain();

    // Noise (Vinyl/Air)
    this.noiseGain = this.ctx.createGain();
    this.noiseGain.gain.value = 0.1;
    this.createNoiseGenerator();

    // Create a simple procedural impulse response for reverb
    const length = this.ctx.sampleRate * 3;
    const impulse = this.ctx.createBuffer(2, length, this.ctx.sampleRate);
    for (let i = 0; i < 2; i++) {
      const channel = impulse.getChannelData(i);
      for (let j = 0; j < length; j++) {
        channel[j] = (Math.random() * 2 - 1) * Math.pow(1 - j / length, 2);
      }
    }
    this.reverb.buffer = impulse;

    // Routing
    // Layers -> Their Gains -> Main Filter -> Master Gain -> Destination
    this.drumGainNode.connect(this.mainFilter);
    this.bassGainNode.connect(this.mainFilter);
    this.chordGainNode.connect(this.mainFilter);
    this.ambienceGainNode.connect(this.mainFilter);
    this.noiseGain.connect(this.mainFilter);

    this.mainFilter.connect(this.masterGain);
    
    // Sends
    this.mainFilter.connect(this.reverbGain);
    this.reverbGain.connect(this.reverb);
    this.reverb.connect(this.masterGain);

    this.mainFilter.connect(this.delayGainNode);
    this.delayGainNode.connect(this.delayNode);
    this.delayNode.connect(this.masterGain);

    this.masterGain.connect(this.ctx.destination);

    this.startScheduler();
  }

  private createNoiseGenerator() {
    if (!this.ctx || !this.noiseGain) return;
    const bufferSize = 2 * this.ctx.sampleRate;
    const buffer = this.ctx.createBuffer(1, bufferSize, this.ctx.sampleRate);
    const output = buffer.getChannelData(0);
    for (let i = 0; i < bufferSize; i++) {
      // Pink-ish noise with some "crackle"
      output[i] = (Math.random() * 2 - 1) * 0.05;
      if (Math.random() > 0.999) output[i] += (Math.random() * 2 - 1) * 0.5;
    }
    const source = this.ctx.createBufferSource();
    source.buffer = buffer;
    source.loop = true;
    source.connect(this.noiseGain);
    source.start();
  }

  updateParams(data: AnalysisData) {
    this.params = data;
    this.applyParams();
  }

  updateGestureParams(x: number | null, y: number | null) {
    if (x !== null && y !== null) {
      this.targetGestureX = x;
      this.targetGestureY = y;
      this.isGestureActive = true;
    } else {
      // Gradually return to neutral
      this.targetGestureX = 0.5;
      this.targetGestureY = 0.5;
      this.isGestureActive = false;
    }
  }

  private applyParams() {
    if (!this.ctx || !this.mainFilter || !this.reverbGain || !this.noiseGain || !this.drumGainNode || !this.bassGainNode || !this.delayGainNode || !this.ambienceGainNode) return;

    const time = this.ctx.currentTime;
    const smoothing = 0.05; // Faster response

    // Smooth gesture values (lerp)
    const lerpFactor = 0.35; // Increased from 0.1
    this.gestureX += (this.targetGestureX - this.gestureX) * lerpFactor;
    this.gestureY += (this.targetGestureY - this.gestureY) * lerpFactor;

    // 1. URBAN -> rhythm and density
    const targetDrumGain = 0.3 + this.params.urban * 0.7;
    const targetBassGain = 0.3 + this.params.urban * 0.6;
    this.drumGainNode.gain.setTargetAtTime(targetDrumGain, time, smoothing);
    this.bassGainNode.gain.setTargetAtTime(targetBassGain, time, smoothing);

    // 2. WATER -> spatial depth + Gesture Y modulation (Distance Perception)
    // Top (0) -> dry/close, Bottom (1) -> far/ambient
    const gestureAmbienceMod = this.gestureY * 2.0; 
    const baseReverb = this.params.water * 0.3;
    const targetReverbWet = baseReverb + gestureAmbienceMod; 
    
    const baseDelay = this.params.water * 0.15;
    const targetDelayMix = baseDelay + gestureAmbienceMod * 0.5;

    this.reverbGain.gain.setTargetAtTime(Math.min(2.0, targetReverbWet), time, smoothing);
    this.delayGainNode.gain.setTargetAtTime(Math.min(1.2, targetDelayMix), time, smoothing);

    // 3. GREEN -> warmth and texture + Gesture X (Brightness) + Gesture Y (Distance Filter)
    // Left (0) -> 200Hz, Right (1) -> 12000Hz
    // Distance filter: as Y increases (bottom), frequency decreases (muffled)
    const minFreq = 200;
    const maxFreq = 12000;
    const baseFreq = minFreq * Math.pow(maxFreq / minFreq, this.gestureX);
    
    // At Y=0 (top), multiplier is 1.0. At Y=1 (bottom), multiplier is 0.15 (muffled).
    const distanceFilterMod = 1.0 - (this.gestureY * 0.85);
    const targetFilterFreq = baseFreq * distanceFilterMod;

    const targetAmbienceGain = 0.1 + this.params.green * 0.4;
    
    this.mainFilter.frequency.setTargetAtTime(Math.max(50, Math.min(20000, targetFilterFreq)), time, smoothing);
    this.noiseGain.gain.setTargetAtTime(targetAmbienceGain * 0.5, time, smoothing);
    this.ambienceGainNode.gain.setTargetAtTime(targetAmbienceGain, time, smoothing);
  }

  getDebugInfo() {
    return {
      x: this.gestureX,
      y: this.gestureY,
      cutoff: this.mainFilter?.frequency.value || 0,
      reverb: this.reverbGain?.gain.value || 0,
      isActive: this.isGestureActive
    };
  }

  private startScheduler() {
    this.nextStepTime = this.ctx!.currentTime;
    this.scheduler();
  }

  private scheduler = () => {
    while (this.nextStepTime < this.ctx!.currentTime + 0.1) {
      if (this.isPlaying) {
        this.playStep(this.step, this.nextStepTime);
        // Apply params more frequently for smooth gesture control
        this.applyParams();
      }
      this.nextStepTime += 60 / this.tempo / 4; // 16th notes
      this.step = (this.step + 1) % 16;
    }
    this.schedulerTimer = setTimeout(this.scheduler, 25);
  };

  private playStep(step: number, time: number) {
    // Steady Lofi Beat
    if (step === 0 || step === 10) {
      this.playKick(time);
    }
    if (step === 4 || step === 12) {
      this.playSnare(time);
    }
    if (step % 2 === 1) {
      this.playHihat(time);
    }

    // Steady Bass
    if (step % 4 === 0) {
      this.playBass(time);
    }

    // Steady Chords
    if (step % 8 === 0) {
      this.playChord(time);
    }

    // Steady Ambience
    if (step === 0) {
      this.playPad(time);
    }
  }

  private playBass(time: number) {
    if (!this.ctx || !this.bassGainNode) return;
    const osc = this.ctx.createOscillator();
    const gain = this.ctx.createGain();
    osc.type = "triangle";
    
    // Simple bass line in F
    const root = 43.65; // F1
    const freq = root * (this.step % 8 < 4 ? 1 : 0.75); // F1 or C1
    osc.frequency.setValueAtTime(freq, time);
    
    osc.connect(gain);
    gain.connect(this.bassGainNode);
    
    gain.gain.setValueAtTime(0, time);
    gain.gain.linearRampToValueAtTime(0.5, time + 0.1);
    gain.gain.exponentialRampToValueAtTime(0.01, time + 0.5);
    
    osc.start(time);
    osc.stop(time + 0.5);
  }

  private playKick(time: number) {
    const osc = this.ctx!.createOscillator();
    const gain = this.ctx!.createGain();
    osc.connect(gain);
    gain.connect(this.drumGainNode!);

    osc.frequency.setValueAtTime(60, time);
    osc.frequency.exponentialRampToValueAtTime(0.01, time + 0.3);
    gain.gain.setValueAtTime(0.4, time);
    gain.gain.exponentialRampToValueAtTime(0.01, time + 0.3);

    osc.start(time);
    osc.stop(time + 0.3);
  }

  private playSnare(time: number) {
    const osc = this.ctx!.createOscillator();
    const gain = this.ctx!.createGain();
    osc.frequency.setValueAtTime(180, time);
    osc.connect(gain);
    
    const noise = this.ctx!.createBufferSource();
    const bufferSize = this.ctx!.sampleRate * 0.1;
    const buffer = this.ctx!.createBuffer(1, bufferSize, this.ctx!.sampleRate);
    const data = buffer.getChannelData(0);
    for (let i = 0; i < bufferSize; i++) data[i] = Math.random() * 2 - 1;
    noise.buffer = buffer;
    
    const noiseGain = this.ctx!.createGain();
    noise.connect(noiseGain);
    noiseGain.connect(gain);
    
    gain.connect(this.drumGainNode!);
    gain.gain.setValueAtTime(0.2, time);
    gain.gain.exponentialRampToValueAtTime(0.01, time + 0.1);

    osc.start(time);
    osc.stop(time + 0.1);
    noise.start(time);
    noise.stop(time + 0.1);
  }

  private playHihat(time: number) {
    const osc = this.ctx!.createOscillator();
    const gain = this.ctx!.createGain();
    const filter = this.ctx!.createBiquadFilter();
    
    osc.type = "square";
    osc.frequency.setValueAtTime(10000, time);
    filter.type = "highpass";
    filter.frequency.setValueAtTime(8000, time);
    
    osc.connect(filter);
    filter.connect(gain);
    gain.connect(this.drumGainNode!);

    gain.gain.setValueAtTime(0.05, time);
    gain.gain.exponentialRampToValueAtTime(0.01, time + 0.05);

    osc.start(time);
    osc.stop(time + 0.05);
  }

  private playChord(time: number) {
    // Lofi chords (Major 7th / Minor 7th)
    const root = 174.61; // F3
    const intervals = [0, 4, 7, 11]; // Maj7
    const notes = intervals;
    
    notes.forEach((interval, i) => {
      const osc = this.ctx!.createOscillator();
      const gain = this.ctx!.createGain();
      osc.type = "triangle";
      osc.frequency.value = root * Math.pow(2, interval / 12);
      
      osc.connect(gain);
      gain.connect(this.chordGainNode!);
      
      const vol = 0.1 / notes.length;
      gain.gain.setValueAtTime(0, time + i * 0.02); // Slight strum
      gain.gain.linearRampToValueAtTime(vol, time + i * 0.02 + 0.1);
      gain.gain.exponentialRampToValueAtTime(0.01, time + 2);
      
      osc.start(time + i * 0.02);
      osc.stop(time + 2);
    });
  }

  private playPad(time: number) {
    const freqs = [220, 329.63, 440, 659.25]; // A3, E4, A4, E5
    freqs.forEach(f => {
      const osc = this.ctx!.createOscillator();
      const gain = this.ctx!.createGain();
      osc.type = "sine";
      osc.frequency.value = f + Math.random() * 2;
      osc.connect(gain);
      gain.connect(this.ambienceGainNode!);
      
      gain.gain.setValueAtTime(0, time);
      gain.gain.linearRampToValueAtTime(0.1, time + 2);
      gain.gain.linearRampToValueAtTime(0, time + 4);
      
      osc.start(time);
      osc.stop(time + 4);
    });
  }

  toggle(playing?: boolean) {
    this.isPlaying = playing ?? !this.isPlaying;
    if (this.isPlaying && this.ctx?.state === "suspended") {
      this.ctx.resume();
    }
  }

  stop() {
    this.isPlaying = false;
    if (this.schedulerTimer) clearTimeout(this.schedulerTimer);
  }
}

// --- Gesture Controller Component ---

const GestureController = ({ onUpdate, isActive }: { onUpdate: (x: number | null, y: number | null) => void, isActive: boolean }) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const handsRef = useRef<any>(null);
  const cameraRef = useRef<any>(null);
  const [isTracking, setIsTracking] = useState(false);
  const [cameraError, setCameraError] = useState<string | null>(null);

  useEffect(() => {
    if (!isActive) {
      if (cameraRef.current) cameraRef.current.stop();
      setCameraError(null);
      return;
    }

    const hands = new Hands({
      locateFile: (file) => {
        return `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`;
      }
    });

    hands.setOptions({
      maxNumHands: 1,
      modelComplexity: 1,
      minDetectionConfidence: 0.5,
      minTrackingConfidence: 0.5
    });

    hands.onResults((results: any) => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      ctx.save();
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      
      if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
        setIsTracking(true);
        const landmarks = results.multiHandLandmarks[0];
        
        // Use palm center (landmark 0 is wrist, 9 is middle finger base)
        const wrist = landmarks[0];
        const middleBase = landmarks[9];
        const centerX = (wrist.x + middleBase.x) / 2;
        const centerY = (wrist.y + middleBase.y) / 2;

        // MediaPipe X is 0-1 (left to right), Y is 0-1 (top to bottom)
        // We want to normalize it. 
        // Note: Webcam is usually mirrored, but MediaPipe might handle it.
        // Let's mirror X for natural feeling
        onUpdate(1 - centerX, centerY);

        // Draw minimal feedback
        ctx.fillStyle = '#62CFB7';
        ctx.shadowBlur = 10;
        ctx.shadowColor = '#62CFB7';
        ctx.beginPath();
        ctx.arc(centerX * canvas.width, centerY * canvas.height, 4, 0, Math.PI * 2);
        ctx.fill();
      } else {
        setIsTracking(false);
        onUpdate(null, null);
      }
      ctx.restore();
    });

    handsRef.current = hands;

    if (videoRef.current) {
      const camera = new Camera(videoRef.current, {
        onFrame: async () => {
          if (videoRef.current && handsRef.current) {
            await handsRef.current.send({ image: videoRef.current });
          }
        },
        width: 160,
        height: 120
      });
      
      camera.start().catch((err) => {
        console.error("Camera start failed:", err);
        setCameraError(err.message || "Failed to access camera");
      });
      
      cameraRef.current = camera;
    }

    return () => {
      if (cameraRef.current) cameraRef.current.stop();
      if (handsRef.current) handsRef.current.close();
    };
  }, [isActive, onUpdate]);

  return (
    <div className="fixed bottom-8 right-8 flex flex-col items-end gap-3 z-50">
      <AnimatePresence>
        {isActive && (
          <motion.div
            initial={{ opacity: 0, y: 20, scale: 0.9 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 20, scale: 0.9 }}
            className="relative group"
          >
            {/* Video Preview */}
            <div className="w-40 h-30 bg-[#0F1512] border border-[#1A2320] overflow-hidden relative">
              <video
                ref={videoRef}
                className="w-full h-full object-cover opacity-40 grayscale scale-x-[-1]"
                playsInline
                muted
              />
              <canvas
                ref={canvasRef}
                width={160}
                height={120}
                className="absolute inset-0 w-full h-full pointer-events-none scale-x-[-1]"
              />
              
              {/* Scanline effect */}
              <div className="absolute inset-0 pointer-events-none bg-gradient-to-b from-transparent via-emerald-500/5 to-transparent h-1/2 animate-scan" />
            </div>

            {/* Status Indicator */}
            <div className="absolute -top-6 right-0 flex items-center gap-2">
              <span className={`text-[8px] font-mono tracking-[0.2em] uppercase ${cameraError ? 'text-red-400' : 'text-[#5A6A66]'}`}>
                {cameraError ? `ERROR: ${cameraError}` : (isTracking ? 'GESTURE_ACTIVE' : 'SEARCHING_HAND...')}
              </span>
              <div className={`w-1 h-1 rounded-full ${cameraError ? 'bg-red-500' : (isTracking ? 'bg-emerald-400 animate-pulse' : 'bg-zinc-600')}`} />
            </div>

            {/* Corner Accents */}
            <div className="absolute top-0 left-0 w-1 h-1 border-t border-l border-[#2A3A36]" />
            <div className="absolute bottom-0 right-0 w-1 h-1 border-b border-r border-[#2A3A36]" />
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default function App() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const mapImageRef = useRef<HTMLImageElement | null>(null);
  const audioEngine = useRef<MapAudioEngine>(new MapAudioEngine());
  
  const [isLoaded, setIsLoaded] = useState(false);
  const [mapImage, setMapImage] = useState<HTMLImageElement | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [isGestureMode, setIsGestureMode] = useState(false);
  const [debugInfo, setDebugInfo] = useState<{x:number, y:number, cutoff:number, reverb:number, isActive:boolean} | null>(null);
  const [error, setError] = useState<string | null>(null);
  
  // Map state
  const [offset, setOffset] = useState<Point>({ x: 0, y: 0 });
  const [zoom, setZoom] = useState(1);
  const [analysis, setAnalysis] = useState<AnalysisData>({ green: 0, water: 0, urban: 0 });
  const [history, setHistory] = useState<AnalysisData[]>([]);
  const [grid, setGrid] = useState<string[]>(new Array(80).fill('#1A2320'));
  
  const isDragging = useRef(false);
  const lastMousePos = useRef<Point>({ x: 0, y: 0 });

  const clampOffset = useCallback((x: number, y: number, z: number, imgOverride?: HTMLImageElement | null) => {
    const img = imgOverride || mapImage || mapImageRef.current;
    if (!img) return { x, y };
    
    const maxOffsetX = VIEWPORT_SIZE / 2 - VIEWPORT_SIZE / (2 * z);
    const minOffsetX = VIEWPORT_SIZE / 2 + VIEWPORT_SIZE / (2 * z) - img.width;
    
    const maxOffsetY = VIEWPORT_SIZE / 2 - VIEWPORT_SIZE / (2 * z);
    const minOffsetY = VIEWPORT_SIZE / 2 + VIEWPORT_SIZE / (2 * z) - img.height;

    return {
      x: Math.min(Math.max(x, minOffsetX), maxOffsetX),
      y: Math.min(Math.max(y, minOffsetY), maxOffsetY)
    };
  }, [mapImage]);

  // Handle Resize
  useEffect(() => {
    const handleResize = () => {
      if (canvasRef.current) {
        canvasRef.current.width = VIEWPORT_SIZE;
        canvasRef.current.height = VIEWPORT_SIZE;
      }
    };
    handleResize();
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  // Initialize Map
  useEffect(() => {
    const img = new Image();
    img.src = MAP_SOURCE;
    
    img.onload = () => {
      mapImageRef.current = img;
      setMapImage(img);
      
      const minZoom = Math.max(VIEWPORT_SIZE / img.width, VIEWPORT_SIZE / img.height);
      setZoom(minZoom);
      
      const initialX = (VIEWPORT_SIZE - img.width) / 2;
      const initialY = (VIEWPORT_SIZE - img.height) / 2;
      
      // Use local clamp logic to avoid dependency loop
      const maxOffsetX = VIEWPORT_SIZE / 2 - VIEWPORT_SIZE / (2 * minZoom);
      const minOffsetX = VIEWPORT_SIZE / 2 + VIEWPORT_SIZE / (2 * minZoom) - img.width;
      const maxOffsetY = VIEWPORT_SIZE / 2 - VIEWPORT_SIZE / (2 * minZoom);
      const minOffsetY = VIEWPORT_SIZE / 2 + VIEWPORT_SIZE / (2 * minZoom) - img.height;

      setOffset({
        x: Math.min(Math.max(initialX, minOffsetX), maxOffsetX),
        y: Math.min(Math.max(initialY, minOffsetY), maxOffsetY)
      });
      
      setIsLoaded(true);
    };
    
    img.onerror = () => {
      setError(`Failed to load ${MAP_SOURCE}. Please ensure the file exists in the project root.`);
    };
  }, []); // Run once on mount

  // Analysis Loop
  useEffect(() => {
    if (!isLoaded) return;

    const interval = setInterval(() => {
      const img = mapImage || mapImageRef.current;
      if (!img) return;

      // Create a temporary canvas for sampling from the SAME image
      const tempCanvas = document.createElement('canvas');
      tempCanvas.width = ANALYSIS_WINDOW_SIZE;
      tempCanvas.height = ANALYSIS_WINDOW_SIZE;
      const tCtx = tempCanvas.getContext("2d", { willReadFrequently: true });
      if (!tCtx) return;

      // Calculate source coordinates for the analysis window
      const halfViewport = VIEWPORT_SIZE / 2;
      const halfWindow = ANALYSIS_WINDOW_SIZE / 2;
      
      const sx = ( (halfViewport - halfWindow) - halfViewport ) / zoom - offset.x + halfViewport;
      const sy = ( (halfViewport - halfWindow) - halfViewport ) / zoom - offset.y + halfViewport;
      
      const sw = ANALYSIS_WINDOW_SIZE / zoom;
      const sh = ANALYSIS_WINDOW_SIZE / zoom;

      try {
        // Draw the relevant part of the image to the temp canvas
        tCtx.drawImage(img, sx, sy, sw, sh, 0, 0, ANALYSIS_WINDOW_SIZE, ANALYSIS_WINDOW_SIZE);
        const imageData = tCtx.getImageData(0, 0, ANALYSIS_WINDOW_SIZE, ANALYSIS_WINDOW_SIZE);
        const pixels = imageData.data;
        
        const GRID_X = 20;
        const GRID_Y = 4;
        const newGrid: string[] = new Array(GRID_X * GRID_Y).fill('#1A2320');
        
        // Classification rules based on user requirements
        const classify = (r: number, g: number, b: number) => {
          // 1. Water (blue)
          if (b > 150 && b > r + 40 && b > g + 40) return { type: 'water', weight: 1 };

          // 2. Green (vegetation)
          if (g > 140 && g > r + 30 && g > b + 30) return { type: 'green', weight: 1 };

          // 3. Urban (low saturation / neutral colors)
          if (Math.abs(r - g) < 40 && Math.abs(g - b) < 40) {
            const brightness = (r + g + b) / 3;
            let weight = 1 - (brightness / 255);
            
            let type: string;
            if (brightness < 40) {
              weight = 1;
              type = 'urban_black';
            } else if (brightness < 120) {
              type = 'urban_high';
            } else {
              type = 'urban_low';
            }
            
            return { type, weight };
          }

          return { type: 'none', weight: 0 };
        };

        // Grid analysis for SpectralMatrix
        for (let gy = 0; gy < GRID_Y; gy++) {
          for (let gx = 0; gx < GRID_X; gx++) {
            const px = Math.floor((gx + 0.5) * (imageData.width / GRID_X));
            const py = Math.floor((gy + 0.5) * (imageData.height / GRID_Y));
            const i = (py * imageData.width + px) * 4;
            
            const r = pixels[i];
            const g = pixels[i+1];
            const b = pixels[i+2];
            
            const { type } = classify(r, g, b);
            const idx = gy * GRID_X + gx;
            
            if (type === 'water') newGrid[idx] = '#208AB8';
            else if (type === 'green') newGrid[idx] = '#4A9B8A';
            else if (type === 'urban_black') newGrid[idx] = '#000000';
            else if (type === 'urban_high') newGrid[idx] = '#3A3A3A';
            else if (type === 'urban_low') newGrid[idx] = '#A0A0A0';
            else newGrid[idx] = '#1A2320';
          }
        }

        // Global analysis for metrics
        const step = 4;
        let greenCount = 0;
        let waterCount = 0;
        let urbanWeightSum = 0;
        let totalValid = 0;

        for (let i = 0; i < pixels.length; i += 4 * step) {
          const r = pixels[i];
          const g = pixels[i+1];
          const b = pixels[i+2];
          
          const { type, weight } = classify(r, g, b);
          if (type !== 'none') {
            totalValid++;
            if (type === 'green') greenCount++;
            else if (type === 'water') waterCount++;
            else urbanWeightSum += weight;
          }
        }

        let rawData = {
          green: totalValid > 0 ? greenCount / totalValid : 0,
          water: totalValid > 0 ? waterCount / totalValid : 0,
          urban: totalValid > 0 ? urbanWeightSum / totalValid : 0,
        };

        // Add subtle system "life" (fluctuations)
        const time = Date.now() * 0.001;
        const drift = (freq: number, amp: number) => Math.sin(time * freq) * amp;
        
        const newData = {
          green: Math.max(0, Math.min(1, rawData.green + drift(1.2, 0.01))),
          water: Math.max(0, Math.min(1, rawData.water + drift(0.8, 0.01))),
          urban: Math.max(0, Math.min(1, rawData.urban + drift(0.5, 0.01))),
        };
        
        setGrid(newGrid);
        setAnalysis(newData);
        setHistory(prev => {
          const next = [...prev, newData];
          return next.slice(-20);
        });
        
        if (isPlaying) {
          audioEngine.current.updateParams(newData);
          setDebugInfo(audioEngine.current.getDebugInfo());
        } else {
          setDebugInfo(null);
        }
      } catch (e) {
        // Bounds errors are expected when dragging near edges
      }
    }, 100);

    return () => clearInterval(interval);
  }, [isLoaded, zoom, offset, isPlaying]);

  // Render Loop
  useEffect(() => {
    let frame: number;
    const render = () => {
      const canvas = canvasRef.current;
      if (!canvas) {
        frame = requestAnimationFrame(render);
        return;
      }
      const ctx = canvas.getContext("2d");
      if (!ctx) {
        frame = requestAnimationFrame(render);
        return;
      }

      ctx.clearRect(0, 0, canvas.width, canvas.height);

      const currentMap = mapImage || mapImageRef.current;
      if (currentMap && isLoaded) {
        // 1. Draw the "Background" (Outside) - Brightness Reduced Only
        ctx.save();
        ctx.translate(VIEWPORT_SIZE / 2, VIEWPORT_SIZE / 2);
        ctx.scale(zoom, zoom);
        ctx.translate(-VIEWPORT_SIZE / 2, -VIEWPORT_SIZE / 2);
        
        ctx.filter = 'brightness(0.75)';
        ctx.drawImage(currentMap, offset.x, offset.y);
        ctx.restore();

        // 2. Draw the "Sampling Zone" (Inside) - Full Reality
        const x1 = (VIEWPORT_SIZE - ANALYSIS_WINDOW_SIZE) / 2;
        const y1 = (VIEWPORT_SIZE - ANALYSIS_WINDOW_SIZE) / 2;
        
        ctx.save();
        ctx.beginPath();
        ctx.rect(x1, y1, ANALYSIS_WINDOW_SIZE, ANALYSIS_WINDOW_SIZE);
        ctx.clip();
        
        ctx.translate(VIEWPORT_SIZE / 2, VIEWPORT_SIZE / 2);
        ctx.scale(zoom, zoom);
        ctx.translate(-VIEWPORT_SIZE / 2, -VIEWPORT_SIZE / 2);
        
        ctx.filter = 'none';
        ctx.drawImage(currentMap, offset.x, offset.y);
        ctx.restore();
      } else if (!isLoaded) {
        // Loading state
        ctx.font = "10px monospace";
        ctx.fillStyle = "rgba(95, 143, 139, 0.5)";
        ctx.textAlign = "center";
        ctx.fillText("INITIALIZING_SOURCE_DATA...", VIEWPORT_SIZE / 2, VIEWPORT_SIZE / 2);
        
        // Animated loading bar
        const barW = 100;
        const barH = 2;
        const bx = (VIEWPORT_SIZE - barW) / 2;
        const by = VIEWPORT_SIZE / 2 + 10;
        ctx.strokeStyle = "rgba(95, 143, 139, 0.2)";
        ctx.strokeRect(bx, by, barW, barH);
        const progress = (Date.now() % 1000) / 1000;
        ctx.fillStyle = "rgba(95, 143, 139, 0.6)";
        ctx.fillRect(bx, by, barW * progress, barH);
      }
      
      // Subtle Overlay with Cutout for Sampling Area (Vignette & Dimming)
      const x1 = (VIEWPORT_SIZE - ANALYSIS_WINDOW_SIZE) / 2;
      const y1 = (VIEWPORT_SIZE - ANALYSIS_WINDOW_SIZE) / 2;
      const time = Date.now();
      
      ctx.save();
      ctx.beginPath();
      // Outer rectangle
      ctx.rect(0, 0, VIEWPORT_SIZE, VIEWPORT_SIZE);
      // Inner rectangle (cutout)
      ctx.moveTo(x1, y1);
      ctx.lineTo(x1, y1 + ANALYSIS_WINDOW_SIZE);
      ctx.lineTo(x1 + ANALYSIS_WINDOW_SIZE, y1 + ANALYSIS_WINDOW_SIZE);
      ctx.lineTo(x1 + ANALYSIS_WINDOW_SIZE, y1);
      ctx.closePath();
      ctx.clip("evenodd");

      // Additional slight dimming for the outside (Pure Black)
      ctx.fillStyle = "rgba(0, 0, 0, 0.4)";
      ctx.fillRect(0, 0, VIEWPORT_SIZE, VIEWPORT_SIZE);
      
      ctx.restore();

      // --- Active Scanning Instrument Effects ---
      
      // 1. Inside Area Processing Effect
      ctx.save();
      ctx.beginPath();
      ctx.rect(x1, y1, ANALYSIS_WINDOW_SIZE, ANALYSIS_WINDOW_SIZE);
      ctx.clip();

      // Internal Scan Line (Reading signal) - Kept very subtle as part of sensor feel
      const scanPos = (time * 0.04) % (ANALYSIS_WINDOW_SIZE * 2);
      if (scanPos < ANALYSIS_WINDOW_SIZE) {
        ctx.strokeStyle = "rgba(95, 143, 139, 0.1)";
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(x1, y1 + scanPos);
        ctx.lineTo(x1 + ANALYSIS_WINDOW_SIZE, y1 + scanPos);
        ctx.stroke();
      }

      ctx.restore();

      // 2. Irregular Borders & Flicker
      const flicker = Math.random() > 0.98 ? 0.4 : 1.0;
      ctx.save();
      ctx.globalAlpha = flicker;
      ctx.strokeStyle = isPlaying ? "#5F8F8B" : "#2A3A36";
      ctx.lineWidth = 1;

      // Jittered dashed border
      ctx.setLineDash([2, 4]);
      const j = () => (Math.random() - 0.5) * 0.4;
      
      const drawJitteredLine = (ax: number, ay: number, bx: number, by: number) => {
        ctx.beginPath();
        ctx.moveTo(ax + j(), ay + j());
        ctx.lineTo(bx + j(), by + j());
        ctx.stroke();
      };

      drawJitteredLine(x1, y1, x1 + ANALYSIS_WINDOW_SIZE, y1);
      drawJitteredLine(x1, y1 + ANALYSIS_WINDOW_SIZE, x1 + ANALYSIS_WINDOW_SIZE, y1 + ANALYSIS_WINDOW_SIZE);
      drawJitteredLine(x1, y1, x1, y1 + ANALYSIS_WINDOW_SIZE);
      drawJitteredLine(x1 + ANALYSIS_WINDOW_SIZE, y1, x1 + ANALYSIS_WINDOW_SIZE, y1 + ANALYSIS_WINDOW_SIZE);

      // 3. Imperfect Corner Brackets
      ctx.setLineDash([]);
      const bSize = 6;
      const bGap = 1;
      
      // Top Left
      ctx.beginPath();
      ctx.moveTo(x1 - bGap + j(), y1 + bSize + j());
      ctx.lineTo(x1 - bGap + j(), y1 - bGap + j());
      ctx.lineTo(x1 + bSize + j(), y1 - bGap + j());
      ctx.stroke();

      // Top Right
      ctx.beginPath();
      ctx.moveTo(x1 + ANALYSIS_WINDOW_SIZE + bGap + j(), y1 + bSize + j());
      ctx.lineTo(x1 + ANALYSIS_WINDOW_SIZE + bGap + j(), y1 - bGap + j());
      ctx.lineTo(x1 + ANALYSIS_WINDOW_SIZE - bSize + j(), y1 - bGap + j());
      ctx.stroke();

      // Bottom Left
      ctx.beginPath();
      ctx.moveTo(x1 - bGap + j(), y1 + ANALYSIS_WINDOW_SIZE - bSize + j());
      ctx.lineTo(x1 - bGap + j(), y1 + ANALYSIS_WINDOW_SIZE + bGap + j());
      ctx.lineTo(x1 + bSize + j(), y1 + ANALYSIS_WINDOW_SIZE + bGap + j());
      ctx.stroke();

      // Bottom Right
      ctx.beginPath();
      ctx.moveTo(x1 + ANALYSIS_WINDOW_SIZE + bGap + j(), y1 + ANALYSIS_WINDOW_SIZE - bSize + j());
      ctx.lineTo(x1 + ANALYSIS_WINDOW_SIZE + bGap + j(), y1 + ANALYSIS_WINDOW_SIZE + bGap + j());
      ctx.lineTo(x1 + ANALYSIS_WINDOW_SIZE - bSize + j(), y1 + ANALYSIS_WINDOW_SIZE + bGap + j());
      ctx.stroke();

      ctx.restore();

      // Visual Reticle (Decorative/Reference only now)
      ctx.strokeStyle = isPlaying ? "rgba(95, 143, 139, 0.3)" : "rgba(42, 58, 54, 0.1)";
      ctx.lineWidth = 1;
      
      // Center Crosshair (Jittered)
      ctx.beginPath();
      ctx.strokeStyle = isPlaying ? "rgba(95, 143, 139, 0.5)" : "rgba(42, 58, 54, 0.15)";
      const cx = VIEWPORT_SIZE / 2;
      const cy = VIEWPORT_SIZE / 2;
      const cj = () => (Math.random() - 0.5) * 0.5;
      
      ctx.moveTo(cx - 15 + cj(), cy + cj());
      ctx.lineTo(cx + 15 + cj(), cy + cj());
      ctx.moveTo(cx + cj(), cy - 15 + cj());
      ctx.lineTo(cx + cj(), cy + 15 + cj());
      ctx.stroke();

      // Center circle (Imperfect)
      ctx.beginPath();
      ctx.arc(cx + cj(), cy + cj(), 4 + cj(), 0, Math.PI * 2);
      ctx.stroke();

      // Viewport Edge Glow (if playing)
      if (isPlaying) {
        ctx.strokeStyle = "rgba(95, 143, 139, 0.15)";
        ctx.strokeRect(1, 1, VIEWPORT_SIZE - 2, VIEWPORT_SIZE - 2);
        
        // Global Scanning Line (Analog feel)
        const scanY = (Date.now() % 6000) / 6000 * VIEWPORT_SIZE;
        ctx.beginPath();
        ctx.strokeStyle = "rgba(95, 143, 139, 0.08)";
        ctx.moveTo(0, scanY + cj());
        ctx.lineTo(VIEWPORT_SIZE, scanY + cj());
        ctx.stroke();
        
        // Occasional horizontal interference line
        if (Math.random() > 0.99) {
          ctx.strokeStyle = "rgba(95, 143, 139, 0.2)";
          const iy = Math.random() * VIEWPORT_SIZE;
          ctx.beginPath();
          ctx.moveTo(0, iy);
          ctx.lineTo(VIEWPORT_SIZE, iy);
          ctx.stroke();
        }
      }
      frame = requestAnimationFrame(render);
    };
    render();
    return () => cancelAnimationFrame(frame);
  }, [offset, zoom, isLoaded, isPlaying, mapImage]);

  // Interaction Handlers
  const startInteraction = useCallback((x: number, y: number) => {
    if (!isLoaded) return;
    isDragging.current = true;
    lastMousePos.current = { x, y };
  }, [isLoaded]);

  const handleInteraction = useCallback((x: number, y: number) => {
    if (!isDragging.current || !isLoaded) return;
    const dx = (x - lastMousePos.current.x) / zoom;
    const dy = (y - lastMousePos.current.y) / zoom;
    
    setOffset(prev => clampOffset(prev.x + dx, prev.y + dy, zoom));
    lastMousePos.current = { x, y };
  }, [isLoaded, zoom, clampOffset]);

  const handleMouseDown = (e: React.MouseEvent) => {
    startInteraction(e.clientX, e.clientY);
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    handleInteraction(e.clientX, e.clientY);
  };

  const handleMouseUp = useCallback(() => {
    isDragging.current = false;
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const handleTouch = (e: TouchEvent) => {
      if (e.touches.length !== 1) return;
      e.preventDefault();
      const touch = e.touches[0];
      const rect = canvas.getBoundingClientRect();
      const x = touch.clientX - rect.left;
      const y = touch.clientY - rect.top;
      
      if (e.type === 'touchstart') {
        startInteraction(x, y);
      } else if (e.type === 'touchmove') {
        handleInteraction(x, y);
      }
    };

    canvas.addEventListener('touchstart', handleTouch, { passive: false });
    canvas.addEventListener('touchmove', handleTouch, { passive: false });
    canvas.addEventListener('touchend', handleMouseUp);

    return () => {
      canvas.removeEventListener('touchstart', handleTouch);
      canvas.removeEventListener('touchmove', handleTouch);
      canvas.removeEventListener('touchend', handleMouseUp);
    };
  }, [startInteraction, handleInteraction, handleMouseUp]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      if (!mapImageRef.current) return;
      
      const img = mapImageRef.current;
      const minZoom = Math.max(VIEWPORT_SIZE / img.width, VIEWPORT_SIZE / img.height);
      const delta = e.deltaY > 0 ? 0.9 : 1.1;
      
      setZoom(prev => {
        const nextZoom = Math.min(Math.max(prev * delta, minZoom), 10);
        // Also clamp offset when zoom changes to ensure coverage
        setOffset(curr => clampOffset(curr.x, curr.y, nextZoom));
        return nextZoom;
      });
    };

    canvas.addEventListener('wheel', onWheel, { passive: false });
    return () => canvas.removeEventListener('wheel', onWheel);
  }, [isLoaded, clampOffset]);

  const toggleAudio = async () => {
    if (!isPlaying) {
      await audioEngine.current.init();
    }
    audioEngine.current.toggle();
    setIsPlaying(!isPlaying);
  };

  const handleGestureUpdate = useCallback((x: number | null, y: number | null) => {
    audioEngine.current.updateGestureParams(x, y);
  }, []);

  return (
    <div className="fixed inset-0 bg-[#0A0F0D] text-[#E3E7EC] font-sans overflow-hidden select-none flex items-center justify-center gap-24 transition-all duration-700">
      {/* Background Dot Pattern */}
      <div 
        className="absolute inset-0 opacity-30 pointer-events-none" 
        style={{ 
          backgroundImage: 'var(--background-dots)', 
          backgroundSize: '32px 32px' 
        }} 
      />

      {/* Left Column: Map */}
      <div className="flex-shrink-0 flex flex-col items-center justify-center">
        <motion.div 
          layout
          className="relative border border-[#1A2320] bg-[#0F1512] overflow-hidden"
          style={{ width: VIEWPORT_SIZE, height: VIEWPORT_SIZE }}
          animate={{ 
            borderColor: '#1A2320'
          }}
          transition={{ type: "spring", stiffness: 100, damping: 20 }}
        >
          <canvas
            ref={canvasRef}
            width={VIEWPORT_SIZE}
            height={VIEWPORT_SIZE}
            className={`w-full h-full ${isLoaded ? 'cursor-move' : 'cursor-wait'}`}
            onMouseDown={handleMouseDown}
            onMouseMove={handleMouseMove}
            onMouseUp={handleMouseUp}
            onMouseLeave={handleMouseUp}
          />

          {/* Loading State */}
          {!isLoaded && !error && (
            <div className="absolute inset-0 flex flex-col items-center justify-center bg-black/40 backdrop-blur-sm z-50">
              <Loader2 className="w-8 h-8 text-zinc-500 animate-spin mb-4" />
              <span className="text-[10px] font-mono tracking-[0.3em] text-zinc-500 uppercase">Loading Berlin Map Data</span>
            </div>
          )}
        </motion.div>
      </div>

      {/* Right Column: Information Panel */}
      {isLoaded && !error && (
        <motion.div 
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          className="w-80 flex flex-col gap-12 items-start z-10"
        >
          {/* Title Section */}
          <div className="flex flex-col gap-1.5">
            <div className="flex items-baseline gap-3">
              <motion.h1 
                initial={{ opacity: 0, y: -10 }}
                animate={{ opacity: 1, y: 0 }}
                className="text-[11px] font-mono tracking-[0.4em] text-[#E3E7EC] uppercase whitespace-nowrap"
              >
                CITY_DRIFT_V3.4 <span className="text-[#8F98A3] tracking-[0.2em] ml-2">BY <a href="https://kirachao.com/" target="_blank" rel="noopener noreferrer" className="hover:text-[#E3E7EC] transition-colors">KIRA CHAO</a></span>
              </motion.h1>
            </div>
            <div className="flex items-center gap-3">
              <motion.span 
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.2 }}
                className="text-[9px] font-mono text-[#5A6A66] uppercase tracking-[0.2em]"
              >
                SYSTEM_OBSERVATION_MODE
              </motion.span>
              <div className="w-1 h-1 rounded-full bg-emerald-500/50 animate-pulse" />
            </div>
          </div>

          {/* Metrics Section */}
          <div className="flex flex-col gap-8 w-full">
            <MetricRow 
              label="URBAN" 
              value={analysis.urban * 100} 
              color="#8E8C9F" 
              history={history.map(h => h.urban)}
            />
            <MetricRow 
              label="WATER" 
              value={analysis.water * 100} 
              color="#28B7F3" 
              history={history.map(h => h.water)}
            />
            <MetricRow 
              label="GREEN" 
              value={analysis.green * 100} 
              color="#62CFB7" 
              history={history.map(h => h.green)}
            />
          </div>

          {/* Pixel Block & Audio Toggle Section */}
          <div className="flex flex-col gap-6 w-full items-start">
            <SpectralMatrix colors={grid} />
            
            <div className="flex flex-col gap-3 w-full">
              <button 
                onClick={toggleAudio}
                className={`
                  relative flex items-center justify-center gap-4 px-8 py-3 
                  border transition-all duration-500 group overflow-hidden w-full
                  ${isPlaying 
                    ? 'bg-emerald-500/5 border-emerald-500/40 text-emerald-400 shadow-[0_0_20px_rgba(16,185,129,0.1)]' 
                    : 'bg-[#0F1512] border-[#2A3A36] text-[#5A6A66] hover:border-[#5F8F8B] hover:text-[#E3E7EC] hover:shadow-[0_0_15px_rgba(95,143,139,0.1)]'
                  }
                `}
              >
                {/* Subtle background scan effect */}
                <motion.div 
                  className="absolute inset-0 bg-gradient-to-r from-transparent via-white/[0.02] to-transparent -translate-x-full"
                  animate={{ x: ['100%', '-100%'] }}
                  transition={{ repeat: Infinity, duration: 3, ease: "linear" }}
                />

                <div className="relative flex items-center justify-center">
                  <AnimatePresence mode="wait">
                    {isPlaying ? (
                      <motion.div
                        key="active"
                        initial={{ scale: 0.8, opacity: 0 }}
                        animate={{ scale: 1, opacity: 1 }}
                        exit={{ scale: 0.8, opacity: 0 }}
                      >
                        <div className="w-1.5 h-1.5 bg-emerald-400 rounded-full shadow-[0_0_8px_#10b981]" />
                      </motion.div>
                    ) : (
                      <motion.div
                        key="inactive"
                        initial={{ scale: 0.8, opacity: 0 }}
                        animate={{ scale: 1, opacity: 1 }}
                        exit={{ scale: 0.8, opacity: 0 }}
                      >
                        <Activity size={12} className="opacity-50 group-hover:opacity-100 transition-opacity" />
                      </motion.div>
                    )}
                  </AnimatePresence>
                </div>

                <span className="text-[10px] font-mono tracking-[0.3em] uppercase relative z-10">
                  {isPlaying ? 'SIGNAL ACTIVE' : 'INITIATE SIGNAL'}
                </span>

                {/* Corner Accents */}
                <div className="absolute top-0 left-0 w-1 h-1 border-t border-l border-inherit" />
                <div className="absolute top-0 right-0 w-1 h-1 border-t border-r border-inherit" />
                <div className="absolute bottom-0 left-0 w-1 h-1 border-b border-l border-inherit" />
                <div className="absolute bottom-0 right-0 w-1 h-1 border-b border-r border-inherit" />
              </button>

              {/* Gesture Mode Toggle */}
              <button 
                onClick={() => setIsGestureMode(!isGestureMode)}
                className={`
                  relative flex items-center justify-center gap-4 px-8 py-2 
                  border transition-all duration-500 group overflow-hidden w-full opacity-80
                  ${isGestureMode 
                    ? 'bg-emerald-500/5 border-emerald-500/20 text-emerald-400/70 shadow-[0_0_10px_rgba(16,185,129,0.05)]' 
                    : 'bg-[#0F1512] border-[#1A2320] text-[#3A4A46] hover:border-[#2A3A36] hover:text-[#5A6A66]'
                  }
                `}
              >
                {/* Subtle background scan effect */}
                <motion.div 
                  className="absolute inset-0 bg-gradient-to-r from-transparent via-white/[0.02] to-transparent -translate-x-full"
                  animate={{ x: ['100%', '-100%'] }}
                  transition={{ repeat: Infinity, duration: 3, ease: "linear" }}
                />

                <div className="relative flex items-center justify-center">
                  <AnimatePresence mode="wait">
                    {isGestureMode ? (
                      <motion.div
                        key="active"
                        initial={{ scale: 0.8, opacity: 0 }}
                        animate={{ scale: 1, opacity: 1 }}
                        exit={{ scale: 0.8, opacity: 0 }}
                      >
                        <div className="w-1 h-1 bg-emerald-400/60 rounded-full shadow-[0_0_4px_rgba(16,185,129,0.4)]" />
                      </motion.div>
                    ) : (
                      <motion.div
                        key="inactive"
                        initial={{ scale: 0.8, opacity: 0 }}
                        animate={{ scale: 1, opacity: 1 }}
                        exit={{ scale: 0.8, opacity: 0 }}
                      >
                        <VideoOff size={10} className="opacity-30 group-hover:opacity-60 transition-opacity" />
                      </motion.div>
                    )}
                  </AnimatePresence>
                </div>

                <div className="relative z-10">
                  <span className="text-[9px] font-mono tracking-[0.2em] uppercase">
                    GESTURE MODE
                  </span>
                </div>

                {/* Corner Accents - Muted */}
                <div className="absolute top-0 left-0 w-1 h-1 border-t border-l border-inherit opacity-30" />
                <div className="absolute top-0 right-0 w-1 h-1 border-t border-r border-inherit opacity-30" />
                <div className="absolute bottom-0 left-0 w-1 h-1 border-b border-l border-inherit opacity-30" />
                <div className="absolute bottom-0 right-0 w-1 h-1 border-b border-r border-inherit opacity-30" />
              </button>
            </div>
            
            {isPlaying ? (
              <motion.div 
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="text-[8px] font-mono text-[#3A4A46] uppercase tracking-[0.1em]"
              >
                LATENCY: {(Math.random() * 20 + 10).toFixed(1)}ms // BUFFER: 1024_SAMPLES
              </motion.div>
            ) : (
              <div className="text-[8px] font-mono text-[#2A3A36] uppercase tracking-[0.1em]">
                STATUS: TERMINATE_SIGNAL // WAITING_FOR_INIT
              </div>
            )}
          </div>
        </motion.div>
      )}

      {/* Error State */}
      {error && (
        <div className="absolute inset-0 flex flex-col items-center justify-center bg-black/80 backdrop-blur-md z-50 p-8 text-center">
          <AlertCircle className="w-12 h-12 text-red-500 mb-4" />
          <h2 className="text-sm font-mono tracking-[0.2em] text-red-400 uppercase mb-2">Critical Error</h2>
          <p className="text-xs font-mono text-zinc-500 max-w-md leading-relaxed">{error}</p>
        </div>
      )}

      {/* Gesture Layer */}
      <GestureController isActive={isGestureMode} onUpdate={handleGestureUpdate} />

      {/* Debug Overlay */}
      <AnimatePresence>
        {isGestureMode && debugInfo && (
          <motion.div 
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -20 }}
            className="fixed top-8 left-8 p-4 bg-black/60 border border-[#1A2320] backdrop-blur-md z-50 flex flex-col gap-2 min-w-[180px]"
          >
            <div className="text-[9px] font-mono text-[#5A6A66] uppercase tracking-[0.2em] mb-1">Gesture_Debug_v1</div>
            <div className="flex justify-between items-center">
              <span className="text-[8px] font-mono text-[#8FA0B2]">HAND_X</span>
              <span className="text-[10px] font-mono text-emerald-400 tabular-nums">{debugInfo.x.toFixed(3)}</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-[8px] font-mono text-[#8FA0B2]">HAND_Y</span>
              <span className="text-[10px] font-mono text-emerald-400 tabular-nums">{debugInfo.y.toFixed(3)}</span>
            </div>
            <div className="h-[1px] bg-[#1A2320] my-1" />
            <div className="flex justify-between items-center">
              <span className="text-[8px] font-mono text-[#8FA0B2]">CUTOFF</span>
              <span className="text-[10px] font-mono text-emerald-400 tabular-nums">{debugInfo.cutoff.toFixed(0)}Hz</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-[8px] font-mono text-[#8FA0B2]">REVERB</span>
              <span className="text-[10px] font-mono text-emerald-400 tabular-nums">{(debugInfo.reverb * 100).toFixed(1)}%</span>
            </div>
            <div className="mt-2 flex items-center gap-2">
              <div className={`w-1.5 h-1.5 rounded-full ${debugInfo.isActive ? 'bg-emerald-400 animate-pulse' : 'bg-zinc-700'}`} />
              <span className="text-[8px] font-mono text-[#5A6A66] uppercase tracking-[0.1em]">
                {debugInfo.isActive ? 'Signal_Locked' : 'Signal_Lost'}
              </span>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
