import React, { useState, useRef, useEffect } from 'react';
import { Mic, Square } from 'lucide-react';

export default function Recorder({ onRecordingComplete, isLoading }) {
  const [isRecording, setIsRecording] = useState(false);
  const [recordingTime, setRecordingTime] = useState(0);
  
  const mediaRecorderRef = useRef(null);
  const chunksRef = useRef([]);
  const timerRef = useRef(null);
  const streamRef = useRef(null);
  
  const audioContextRef = useRef(null);
  const analyserRef = useRef(null);
  const dataArrayRef = useRef(null);
  const sourceRef = useRef(null);
  const requestRef = useRef(null);
  const canvasRef = useRef(null);

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      chunksRef.current = [];

      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) {
          chunksRef.current.push(e.data);
        }
      };

      mediaRecorder.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: 'audio/webm' });
        const url = URL.createObjectURL(blob);
        onRecordingComplete({ blob, url, name: 'recorded-audio.webm' });
        stream.getTracks().forEach(track => track.stop());
        streamRef.current = null;
        stopVisualizer();
      };

      mediaRecorder.start();
      setIsRecording(true);
      setRecordingTime(0);
      
      timerRef.current = setInterval(() => {
        setRecordingTime(prev => prev + 1);
      }, 1000);
      
    } catch (err) {
      console.error(err);
      alert('Microphone access denied or not available.');
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      clearInterval(timerRef.current);
    }
  };

  useEffect(() => {
    if (isRecording && streamRef.current) {
      startVisualizer(streamRef.current);
    }
  }, [isRecording]);

  const startVisualizer = (stream) => {
    const audioContext = new (window.AudioContext || window.webkitAudioContext)();
    const analyser = audioContext.createAnalyser();
    const source = audioContext.createMediaStreamSource(stream);
    
    source.connect(analyser);
    analyser.fftSize = 256;
    
    const bufferLength = analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);
    
    audioContextRef.current = audioContext;
    analyserRef.current = analyser;
    dataArrayRef.current = dataArray;
    sourceRef.current = source;
    
    drawVisualizer();
  };
  
  const stopVisualizer = () => {
    if (requestRef.current) cancelAnimationFrame(requestRef.current);
    if (audioContextRef.current?.state !== 'closed' && audioContextRef.current) {
      audioContextRef.current.close().catch(console.error);
    }
  };

  const drawVisualizer = () => {
    if (!canvasRef.current || !analyserRef.current) return;
    
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;
    
    analyserRef.current.getByteFrequencyData(dataArrayRef.current);
    
    ctx.clearRect(0, 0, width, height);
    
    const barWidth = (width / dataArrayRef.current.length) * 2.5;
    let x = 0;
    
    for(let i = 0; i < dataArrayRef.current.length; i++) {
      const barHeight = (dataArrayRef.current[i] / 255) * height;
      
      ctx.fillStyle = '#F4C430'; 
      ctx.fillRect(x, height - barHeight, barWidth, barHeight);
      
      x += barWidth + 1;
    }
    
    requestRef.current = requestAnimationFrame(drawVisualizer);
  };
  
  useEffect(() => {
    return () => {
      clearInterval(timerRef.current);
      stopVisualizer();
      if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
        mediaRecorderRef.current.stop();
      }
    };
  }, []);

  const formatTime = (seconds) => {
    const m = Math.floor(seconds / 60).toString().padStart(2, '0');
    const s = (seconds % 60).toString().padStart(2, '0');
    return `${m}:${s}`;
  };

  return (
    <div className="flex flex-col items-center justify-center h-full w-full py-8">
      {isRecording ? (
        <div className="flex flex-col items-center w-full max-w-sm mx-auto">
          <div className="flex items-center gap-2 mb-4 font-mono text-sm text-red-600">
            <span className="w-2 h-2 rounded-full bg-red-600 animate-pulse"></span>
            RECORDING {formatTime(recordingTime)}
          </div>
          <canvas ref={canvasRef} className="w-full h-16 mb-6 rounded bg-black/5" width={300} height={64}></canvas>
          <button 
            onClick={stopRecording}
            aria-label="Stop recording"
            className="flex items-center gap-2 px-6 py-2 border-2 border-ink text-ink font-mono text-sm uppercase tracking-wider hover:bg-ink hover:text-paper focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ink transition-colors"
          >
            <Square className="w-4 h-4 fill-current" />
            Stop
          </button>
        </div>
      ) : (
        <button
          onClick={startRecording}
          disabled={isLoading}
          aria-label="Start recording"
          className="flex flex-col items-center gap-4 text-ink hover:text-highlight transition-colors disabled:opacity-50 disabled:cursor-not-allowed group focus-visible:outline-none"
        >
          <div className="w-16 h-16 flex items-center justify-center border-2 border-ink rounded-full group-hover:border-highlight group-hover:bg-highlight/10 group-focus-visible:ring-2 group-focus-visible:ring-ink group-focus-visible:ring-offset-2">
            <Mic className="w-6 h-6" />
          </div>
          <span className="font-mono text-sm uppercase tracking-widest text-ink group-hover:text-ink-light">Record Audio</span>
        </button>
      )}
    </div>
  );
}
