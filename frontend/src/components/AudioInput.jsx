import React, { useState } from 'react';
import { Upload, Play, Trash2, FileAudio } from 'lucide-react';
import Recorder from './Recorder';

export default function AudioInput({ onAudioSubmit, onAudioClear, isLoading }) {
  const [activeTab, setActiveTab] = useState('record');
  const [selectedAudio, setSelectedAudio] = useState(null);

  const handleFileUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      if (!file.type.startsWith('audio/')) {
        alert('Please upload an audio file');
        return;
      }
      const url = URL.createObjectURL(file);
      setSelectedAudio({ blob: file, url, name: file.name });
    }
  };

  const clearAudio = () => {
    if (selectedAudio?.url) {
      URL.revokeObjectURL(selectedAudio.url);
    }
    setSelectedAudio(null);
    if (onAudioClear) {
      onAudioClear();
    }
  };

  const handleSubmit = () => {
    if (selectedAudio) {
      onAudioSubmit(selectedAudio.blob, selectedAudio.name, selectedAudio.url);
    }
  };

  const handleRecordingComplete = (audioData) => {
    setSelectedAudio(audioData);
  };

  return (
    <div className="w-full bg-paper border-2 border-ink shadow-[4px_4px_0_0_#1E2530] p-6 mb-8 relative">
      
      <div className="absolute top-0 left-0 right-0 h-1.5 bg-ink" />
      
      <div className="flex border-b-2 border-ink mb-6">
        <button
          onClick={() => setActiveTab('upload')}
          disabled={isLoading || selectedAudio}
          aria-label="Upload Audio Tab"
          className={`flex-1 sm:flex-none px-6 py-2 font-mono text-sm uppercase tracking-wider focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-highlight transition-colors ${
            activeTab === 'upload' ? 'bg-ink text-paper' : 'text-ink hover:bg-black/5'
          } ${selectedAudio ? 'opacity-50 cursor-not-allowed' : ''}`}
        >
          Upload
        </button>
        <button
          onClick={() => setActiveTab('record')}
          disabled={isLoading || selectedAudio}
          aria-label="Record Audio Tab"
          className={`flex-1 sm:flex-none px-6 py-2 font-mono text-sm uppercase tracking-wider focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-highlight transition-colors ${
            activeTab === 'record' ? 'bg-ink text-paper' : 'text-ink hover:bg-black/5'
          } ${selectedAudio ? 'opacity-50 cursor-not-allowed' : ''}`}
        >
          Record
        </button>
      </div>

      <div className="min-h-[180px] flex items-center justify-center bg-white border border-ink/20 p-6">
        {selectedAudio ? (
          <div className="w-full max-w-md mx-auto">
            <div className="flex items-center gap-4 mb-6 p-4 bg-paper border border-ink/20">
              <FileAudio className="w-6 h-6 text-ink/70 flex-shrink-0" />
              <div className="flex-1 min-w-0">
                <p className="font-mono text-sm text-ink truncate font-medium">{selectedAudio.name}</p>
              </div>
              <button 
                onClick={clearAudio} 
                disabled={isLoading}
                aria-label="Remove audio"
                className="p-2 text-ink hover:text-red-600 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ink"
              >
                <Trash2 className="w-5 h-5" />
              </button>
            </div>
            
            <button
              onClick={handleSubmit}
              disabled={isLoading}
              className="w-full flex items-center justify-center gap-2 py-3 bg-highlight text-ink border-2 border-ink font-mono text-sm uppercase tracking-widest hover:bg-highlight/90 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ink focus-visible:ring-offset-2 transition-transform active:translate-y-[2px] shadow-[4px_4px_0_0_#1E2530] active:shadow-[2px_2px_0_0_#1E2530] disabled:opacity-70 disabled:active:translate-y-0 disabled:active:shadow-[4px_4px_0_0_#1E2530]"
            >
              {isLoading ? (
                'ANALYZING...'
              ) : (
                <>
                  <Play className="w-4 h-4 fill-current" />
                  TRANSCRIBE
                </>
              )}
            </button>
          </div>
        ) : (
          <div className="w-full h-full">
            {activeTab === 'upload' ? (
               <label className="flex flex-col items-center justify-center cursor-pointer group p-12 focus-within:ring-2 focus-within:ring-ink focus-within:ring-offset-2 w-full h-full border-2 border-dashed border-ink/20 hover:border-ink/50 bg-paper/50 hover:bg-paper transition-all">
                 <input type="file" accept="audio/*" onChange={handleFileUpload} className="sr-only" disabled={isLoading} />
                 <Upload className="w-8 h-8 text-ink/50 group-hover:text-ink mb-4 transition-colors" />
                 <span className="font-mono text-sm uppercase tracking-widest text-ink/70 group-hover:text-ink">Choose audio file</span>
               </label>
            ) : (
               <Recorder onRecordingComplete={handleRecordingComplete} isLoading={isLoading} />
            )}
          </div>
        )}
      </div>
    </div>
  );
}
