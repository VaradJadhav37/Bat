import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

const BatteryCell = ({ level = 50, label = "Cell 1" }) => {
  const containerStyle = {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    gap: '20px',
    width: '100%',
  };

  const batteryBodyStyle = {
    width: '60px',
    height: '110px',
    border: '3px solid #3e3e5e',
    borderRadius: '12px',
    position: 'relative',
    backgroundColor: '#0f0f1a',
    padding: '3px',
    boxSizing: 'border-box'
  };

  const terminalStyle = {
    width: '20px',
    height: '8px',
    backgroundColor: '#3e3e5e',
    position: 'absolute',
    top: '-11px',
    left: '50%',
    transform: 'translateX(-50%)',
    borderRadius: '4px 4px 0 0'
  };

  const fillContainerStyle = {
    width: '100%',
    height: '100%',
    borderRadius: '8px',
    overflow: 'hidden',
    position: 'relative',
    backgroundColor: 'transparent'
  };

  const fillLevelStyle = {
    position: 'absolute',
    bottom: 0,
    left: 0,
    width: '100%',
    height: `${level}%`,
    background: 'linear-gradient(to bottom, #2F6B3F, #1E4D2B)',
    transition: 'height 1s ease-in-out'
  };

  const waveOverlayStyle = {
    position: 'absolute',
    top: '-15px',
    left: 0,
    width: '200%',
    height: '20px',
    fill: '#2F6B3F',
    animation: 'waveHorizontal 3s infinite linear'
  };

  return (
    <div style={containerStyle}>
      <style>{`
        @keyframes waveHorizontal {
          0% { transform: translateX(0); }
          100% { transform: translateX(-50%); }
        }
      `}</style>
      <div style={batteryBodyStyle}>
        <div style={terminalStyle}></div>
        <div style={fillContainerStyle}>
          <div style={fillLevelStyle}>
            <svg style={waveOverlayStyle} viewBox="0 0 100 20" preserveAspectRatio="none">
              <path d="M0 10 Q 12.5 0 25 10 T 50 10 T 75 10 T 100 10 V 20 H 0 Z" />
            </svg>
          </div>
        </div>
      </div>
      <h2 className="font-headline-md text-headline-md text-surface-bright mt-2">{label}</h2>
    </div>
  );
};

const API_BASE = import.meta.env.VITE_API_BASE_URL || 'https://bat-tbps.onrender.com';

function App() {
  const [loading, setLoading] = useState(false);
  const [testSamples, setTestSamples] = useState([]);
  const [selectedSampleIdx, setSelectedSampleIdx] = useState(0);
  
  // Results
  const [prediction, setPrediction] = useState(null);
  const [bayesian, setBayesian] = useState(null);
  const [metrics, setMetrics] = useState(null);
  
  const fileInputRef = useRef(null);

  useEffect(() => {
    // Load test samples on mount
    axios.get(`${API_BASE}/samples`).then(res => {
      if(res.data && res.data.length > 0) {
        setTestSamples(res.data);
        setSelectedSampleIdx(0);
        handlePredict(res.data[0]);
      }
    }).catch(err => console.error("Error loading samples", err));
    
    // Load Bayesian & metrics
    axios.get(`${API_BASE}/bayesian`).then(res => setBayesian(res.data)).catch(e => console.warn(e));
    axios.get(`${API_BASE}/metrics`).then(res => setMetrics(res.data)).catch(e => console.warn(e));
  }, []);

  useEffect(() => {
    const cell = getCellId();
    if(cell && cell !== 'Unknown Cell') {
      axios.get(`${API_BASE}/bayesian?cell=${cell}`).then(res => setBayesian(res.data)).catch(e => console.warn(e));
    }
  }, [selectedSampleIdx, testSamples]);

  const handlePredict = async (sample) => {
    if(!sample) return;
    setLoading(true);
    try {
      const res = await axios.post(`${API_BASE}/predict`, sample, {
        headers: { 'Content-Type': 'application/json' }
      });
      setPrediction(res.data);
    } catch(err) {
      console.error(err);
      alert("Error running prediction.");
    }
    setLoading(false);
  };

  const handleFileUpload = async (file) => {
    if(!file) return;
    setLoading(true);
    const formData = new FormData();
    formData.append('file', file);
    try {
      const res = await axios.post(`${API_BASE}/upload`, formData);
      const samples = res.data;
      if(samples.length > 0) {
        setTestSamples(samples);
        setSelectedSampleIdx(0);
        handlePredict(samples[0]);
      }
    } catch(err) {
      console.error(err);
      alert("Error processing file.");
    }
    setLoading(false);
  };

  const currentSample = testSamples[selectedSampleIdx];
  
  // Render helpers
  const pct = (val) => val != null ? (val * 100).toFixed(1) + '%' : '--';
  const getCellId = () => prediction?.cell || currentSample?.cell || 'Unknown Cell';
  
  const sohVal = prediction?.ensemble?.soh ?? prediction?.idtcn?.soh ?? prediction?.wavenet?.soh ?? null;
  const rulVal = prediction?.ensemble?.rul_cycles ?? prediction?.idtcn?.rul_cycles ?? null;
  const rulNorm = prediction?.ensemble?.rul_norm ?? prediction?.idtcn?.rul_norm ?? null;
  const nPoints = prediction?.n_points ?? currentSample?.v?.length ?? 0;

  return (
    <div className="bg-surface font-body-md text-on-surface h-screen flex overflow-hidden antialiased">
      {/* SideNavBar */}
      <nav className="bg-[#F5F3EE] dark:bg-[#1A1A1A] font-serif text-sm antialiased border-r border-slate-200 dark:border-slate-800 flat no shadows fixed left-0 top-0 h-full flex-col p-6 gap-8 hidden md:flex z-50 w-64">
        <div className="flex items-center gap-4 px-2">
          <div className="w-10 h-10 rounded-full bg-surface-container-high overflow-hidden shrink-0 flex items-center justify-center text-xl">
             🔋
          </div>
          <div>
            <h1 className="text-xl font-black text-slate-900 dark:text-slate-50 leading-tight">Enerlytics</h1>
            <p className="text-slate-500 text-xs">Battery Prognostics</p>
          </div>
        </div>
        <ul className="flex flex-col gap-2 flex-grow mt-4">
          <li>
            <a className="bg-amber-500/10 text-amber-700 dark:text-amber-400 rounded-full font-bold px-4 py-2 flex items-center gap-3 w-full" href="#">
              <span className="material-symbols-outlined">dashboard</span>
              <span>Overview</span>
            </a>
          </li>
          <li>
            <a className="hover:bg-amber-500/5 text-slate-600 dark:text-slate-400 rounded-full font-bold px-4 py-2 flex items-center gap-3 w-full cursor-pointer" onClick={() => fileInputRef.current?.click()}>
              <span className="material-symbols-outlined">upload_file</span>
              <span>Upload</span>
            </a>
            <input type="file" ref={fileInputRef} className="hidden" accept=".pkl" onChange={(e) => handleFileUpload(e.target.files[0])} />
          </li>
        </ul>
        <button 
          onClick={() => handlePredict(currentSample)}
          disabled={loading || !currentSample}
          className="mt-auto bg-primary text-on-primary rounded-full py-3 px-6 font-bold hover:opacity-90 transition-opacity flex items-center justify-center gap-2 disabled:opacity-50">
          <span>{loading ? 'Running...' : 'Run Model'}</span>
          <span className="material-symbols-outlined text-sm">play_arrow</span>
        </button>
      </nav>

      {/* Mobile Header */}
      <header className="md:hidden bg-surface border-b border-surface-variant p-4 flex items-center justify-between fixed top-0 left-0 w-full z-[100]">
        <div className="flex items-center gap-2">
          <span className="text-2xl">🔋</span>
          <h1 className="font-bold text-lg">Enerlytics</h1>
        </div>
        <button 
          onClick={() => fileInputRef.current?.click()}
          className="bg-primary text-on-primary rounded-full p-2 flex items-center justify-center">
          <span className="material-symbols-outlined">upload_file</span>
        </button>
      </header>

      {/* Main Wrapper */}
      <div className="flex-1 flex flex-col md:ml-64 relative h-screen overflow-hidden pt-[64px] md:pt-0">
        <main className="flex-1 overflow-y-auto p-4 md:p-8 bg-surface-container-low flex flex-col gap-6">
          <div className="flex flex-col lg:flex-row gap-gutter w-full">
          {/* Left Sidebar: Cell Profile Card */}
          <aside className="w-full lg:w-80 shrink-0 flex flex-col md:flex-row lg:flex-col gap-6">
            <div className="bg-primary-container text-on-primary rounded-custom p-6 shadow-lg flex flex-col justify-between gap-6 flex-1 min-h-[300px]">
                  <div className="flex flex-col items-center gap-4">
                    <p className="font-label-caps text-label-caps text-surface-dim uppercase tracking-widest">Cell Profile</p>
                    <BatteryCell level={50} label={getCellId()} />
                  </div>
                {testSamples.length > 0 && (
                  <select 
                    className="mt-2 w-full bg-inverse-surface border border-surface-tint/30 text-surface-bright rounded p-2 text-sm"
                    value={selectedSampleIdx}
                    onChange={(e) => {
                      setSelectedSampleIdx(e.target.value);
                      setPrediction(null);
                    }}
                  >
                    {testSamples.map((s, i) => (
                      <option key={i} value={i}>{s.cell} / {s.cycle} (GT SoH:{pct(s.soh_true)})</option>
                    ))}
                  </select>
                )}
              </div>

            {/* Performance Metrics Card */}
            {metrics && (
              <div className="bg-surface rounded-custom p-8 border border-surface-variant shadow-sm flex flex-col h-[340px]">
                 <h3 className="font-headline-md text-2xl text-on-surface mb-8">Model Performance</h3>
                 <div className="flex-1 overflow-hidden">
                   <table className="w-full text-sm text-left h-full">
                     <thead>
                       <tr className="border-b border-surface-variant text-surface-dim">
                         <th className="py-2">Model</th>
                         <th className="py-2">SOH R²</th>
                         <th className="py-2">RUL R²</th>
                       </tr>
                     </thead>
                     <tbody className="flex-1">
                       <tr className="border-b border-surface-variant/50">
                         <td className="py-4 font-bold text-on-surface-variant">ID-TCN</td>
                         <td className="py-4">{metrics.idtcn?.soh_r2?.toFixed(3)}</td>
                         <td className="py-4">{metrics.idtcn?.rul_r2?.toFixed(3)}</td>
                       </tr>
                       <tr>
                         <td className="py-4 font-bold text-on-surface-variant">WaveNet</td>
                         <td className="py-4">{metrics.wavenet?.soh_r2?.toFixed(3)}</td>
                         <td className="py-4">{metrics.wavenet?.rul_r2?.toFixed(3)}</td>
                       </tr>
                     </tbody>
                   </table>
                 </div>
              </div>
            )}
          </aside>

          {/* Main Content Area */}
          <div className="flex-1 flex flex-col gap-6 max-w-container-max">
            
            {/* Top Row: Stat Pills */}
            <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-gutter">
              <div className="bg-surface rounded-full border border-surface-variant flex items-center justify-between px-8 py-5 shadow-sm">
                <span className="font-label-caps text-label-caps text-on-surface-variant uppercase tracking-widest text-[10px] sm:text-xs">ID-TCN SoH</span>
                <span className="font-metric-lg text-metric-lg text-on-surface">{pct(prediction?.idtcn?.soh)}</span>
              </div>
              <div className="bg-surface rounded-full border border-surface-variant flex items-center justify-between px-8 py-5 shadow-sm">
                <span className="font-label-caps text-label-caps text-on-surface-variant uppercase tracking-widest text-[10px] sm:text-xs">WaveNet SoH</span>
                <span className="font-metric-lg text-metric-lg text-on-tertiary-container">{pct(prediction?.wavenet?.soh)}</span>
              </div>
              <div className="bg-surface rounded-full border border-surface-variant flex items-center justify-between px-8 py-5 shadow-sm col-span-1 sm:col-span-2 md:col-span-1">
                <span className="font-label-caps text-label-caps text-on-surface-variant uppercase tracking-widest text-[10px] sm:text-xs">Ensemble RUL</span>
                <span className="font-metric-lg text-metric-lg text-on-surface">{rulVal ?? '--'} cyc</span>
              </div>
            </div>

            {/* Middle Row: 3 Cards */}
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-gutter min-h-[500px] lg:h-64">
              
              {/* Card A: SoH Progress Bar Chart */}
              <div className="bg-primary-container text-on-primary rounded-custom p-6 shadow-lg flex flex-col relative overflow-hidden group">
                <div className="flex justify-between items-start mb-6 z-10">
                  <h3 className="font-headline-md text-xl text-surface-bright">SoH Prediction</h3>
                  <span className="material-symbols-outlined text-tertiary-fixed-dim text-sm">monitoring</span>
                </div>
                <div className="flex-1 flex items-end justify-center gap-2 z-10 pt-4 border-b border-surface-tint/30 pb-2 h-full">
                  <div className="w-full flex flex-col items-center gap-2 group/bar h-full justify-end">
                    <div className="w-full bg-tertiary-fixed-dim rounded-t-sm relative shadow-[0_0_10px_rgba(255,185,76,0.3)] transition-all duration-1000 flex items-start justify-center" style={{height: `${(sohVal || 0) * 100}%`}}>
                       {sohVal && <span className="text-primary font-black text-xs mt-2">{pct(sohVal)}</span>}
                    </div>
                    <span className="font-metric-sm text-[10px] text-tertiary-fixed-dim font-bold">PREDICTED</span>
                  </div>
                </div>
              </div>

              {/* Card B: RUL Tracker Circular Countdown */}
              <div className="bg-surface text-on-surface rounded-custom p-6 border border-surface-variant shadow-sm flex flex-col items-center justify-center relative">
                <h3 className="absolute top-6 left-6 font-headline-md text-xl text-on-surface">RUL Tracker</h3>
                <div className="relative w-40 h-40 flex items-center justify-center mt-6">
                  <svg className="absolute inset-0 w-full h-full transform -rotate-90" viewBox="0 0 100 100">
                    <circle className="text-surface-container-high" cx="50" cy="50" fill="none" r="45" stroke="currentColor" strokeWidth="2"></circle>
                    <circle className="text-on-tertiary-container transition-all duration-1000 ease-out" cx="50" cy="50" fill="none" r="45" stroke="currentColor" strokeDasharray="283" strokeDashoffset={283 - (283 * (rulNorm || 0))} strokeWidth="4"></circle>
                  </svg>
                  <div className="flex flex-col items-center text-center z-10 bg-surface w-32 h-32 rounded-full justify-center shadow-[inset_0_2px_10px_rgba(0,0,0,0.05)] border border-surface-variant/50">
                    <span className="font-metric-lg text-3xl text-on-tertiary-container">{rulVal ?? '--'}</span>
                    <span className="font-label-caps text-[10px] text-on-surface-variant uppercase tracking-widest mt-1">Cycles</span>
                  </div>
                </div>
              </div>

              {/* Card C: Optimization Status (Current Sample Params) */}
              <div className="bg-surface text-on-surface rounded-custom p-6 border border-surface-variant shadow-sm flex flex-col justify-between">
                <div className="flex justify-between items-start mb-4">
                  <h3 className="font-headline-md text-xl text-on-surface">Sample Physics</h3>
                  <span className="material-symbols-outlined text-on-surface-variant text-sm">tune</span>
                </div>
                <div className="flex flex-col gap-5 flex-1 justify-end">
                  <div className="flex flex-col gap-2">
                    <div className="flex justify-between items-end">
                      <span className="font-label-caps text-xs text-on-surface-variant">Mean Voltage</span>
                      <span className="font-metric-sm text-xs text-on-surface">{prediction?.current_phys?.voltage?.toFixed(4) ?? '--'} V</span>
                    </div>
                    <div className="h-1.5 w-full bg-surface-container-high rounded-full overflow-hidden">
                      <div className="h-full bg-on-tertiary-container rounded-full transition-all" style={{width: `${Math.min(100, Math.max(0, ((prediction?.current_phys?.voltage || 3.5)-3)/(4.2-3)*100))}%`}}></div>
                    </div>
                  </div>
                  <div className="flex flex-col gap-2">
                    <div className="flex justify-between items-end">
                      <span className="font-label-caps text-xs text-on-surface-variant">Mean Current</span>
                      <span className="font-metric-sm text-xs text-on-surface">{prediction?.current_phys?.current?.toFixed(4) ?? '--'} A</span>
                    </div>
                    <div className="h-1.5 w-full bg-surface-container-high rounded-full overflow-hidden">
                      <div className="h-full bg-on-primary-container rounded-full transition-all" style={{width: `${Math.min(100, Math.max(0, (Math.abs(prediction?.current_phys?.current || 0)/2)*100))}%`}}></div>
                    </div>
                  </div>
                  <div className="flex flex-col gap-2">
                    <div className="flex justify-between items-end">
                      <span className="font-label-caps text-xs text-on-surface-variant">DoD</span>
                      <span className="font-metric-sm text-xs text-on-surface">{prediction?.current_phys?.dod?.toFixed(2) ?? '--'}</span>
                    </div>
                    <div className="h-1.5 w-full bg-surface-container-high rounded-full overflow-hidden">
                      <div className="h-full bg-outline rounded-full transition-all" style={{width: `${Math.min(100, ((prediction?.current_phys?.dod || 0)/1000)*100)}%`}}></div>
                    </div>
                  </div>
                </div>
              </div>

            </div>

            {/* Bayesian Optimization Row */}
            {bayesian && (
              <div className="bg-surface text-on-surface rounded-custom p-8 border border-surface-variant shadow-sm min-h-[340px]">
                <div className="flex items-center gap-4 mb-8">
                  <h3 className="font-headline-md text-xl md:text-2xl text-on-surface">Bayesian Optimization <span className="text-surface-tint font-light">— Optimal Charging Parameters</span></h3>
                </div>
                
                <div className="grid grid-cols-1 md:grid-cols-2 gap-gutter">
                  {['idtcn', 'wavenet'].map(k => {
                    const b = bayesian[k];
                    return (
                      <div key={k}>
                        <h4 className="font-bold text-surface-tint mb-3">{k === 'idtcn' ? 'ID-TCN' : 'WaveNet-CNN'} Result</h4>
                        <table className="w-full text-sm text-left">
                          <thead>
                            <tr className="border-b border-surface-variant text-surface-dim">
                              <th className="py-2">Parameter</th>
                              <th className="py-2">Optimal Value</th>
                              <th className="py-2">Current Sample</th>
                            </tr>
                          </thead>
                          <tbody>
                            <tr className="border-b border-surface-variant/50">
                              <td className="py-2">⚡ Voltage</td>
                              <td className="py-2"><strong>{b.best_voltage.toFixed(4)} V</strong></td>
                              <td className="py-2">{prediction?.current_phys ? `${prediction.current_phys.voltage.toFixed(4)} V` : '—'}</td>
                            </tr>
                            <tr className="border-b border-surface-variant/50">
                              <td className="py-2">🔌 Current</td>
                              <td className="py-2"><strong>{b.best_current.toFixed(4)} A</strong></td>
                              <td className="py-2">{prediction?.current_phys ? `${prediction.current_phys.current.toFixed(4)} A` : '—'}</td>
                            </tr>
                            <tr className="border-b border-surface-variant/50">
                              <td className="py-2">📉 Depth of Discharge</td>
                              <td className="py-2"><strong>{b.best_dod.toFixed(4)}</strong></td>
                              <td className="py-2">{prediction?.current_phys ? prediction.current_phys.dod.toFixed(4) : '—'}</td>
                            </tr>
                            <tr>
                              <td className="py-2">🔮 Predicted Max RUL</td>
                              <td className="py-2"><strong className="text-tertiary-fixed-dim">{b.best_rul_cycles ? `${b.best_rul_cycles} cyc` : b.best_rul.toFixed(4)}</strong></td>
                              <td className="py-2">{prediction ? (prediction[k]?.rul_cycles ? `${prediction[k].rul_cycles} cyc` : prediction[k]?.rul_norm?.toFixed(4)) : '—'}</td>
                            </tr>
                          </tbody>
                        </table>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}

            </div> {/* End of Main Content Area flex-col */}
          </div> {/* End of Top Flex Row (Sidebar + Main) */}
            
            {/* Training Curves Row - Full Width under sidebar */}
            {metrics && (
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-gutter min-h-[400px] w-full mt-2">
                {['idtcn', 'wavenet'].map(k => {
                  const m = metrics[k];
                  const data = m.history_loss.map((l, i) => ({
                    epoch: i + 1,
                    loss: l,
                    soh: m.history_soh_val[i],
                    rul: m.history_rul_val[i]
                  }));
                  return (
                    <div key={k} className="bg-primary-container text-on-primary rounded-custom p-6 shadow-lg flex flex-col h-[400px] overflow-hidden">
                      <h3 className="font-label-caps text-label-caps text-surface-bright uppercase tracking-widest mb-6">{k === 'idtcn' ? 'ID-TCN' : 'WaveNet'} Training Curves</h3>
                      <div className="flex-1 w-full min-h-0">
                        <ResponsiveContainer width="100%" height="100%">
                          <LineChart data={data}>
                            <CartesianGrid strokeDasharray="3 3" stroke="#2c2c2c" vertical={false} />
                            <XAxis dataKey="epoch" stroke="#8d9191" fontSize={10} tickLine={false} axisLine={false} />
                            <YAxis stroke="#8d9191" fontSize={10} tickLine={false} axisLine={false} />
                            <Tooltip 
                              contentStyle={{ backgroundColor: '#1b1c1c', borderRadius: '12px', border: '1px solid #2c2c2c', boxShadow: '0 4px 20px rgba(0,0,0,0.5)', color: '#fff' }}
                              itemStyle={{ color: '#fff' }}
                            />
                            <Line type="monotone" dataKey="loss" stroke="#3f6190" strokeWidth={2} dot={false} name="Train Loss" />
                            <Line type="monotone" dataKey="soh" stroke="#00696d" strokeWidth={2} dot={false} name="SOH MAE" />
                            <Line type="monotone" dataKey="rul" stroke="#ba1a1a" strokeWidth={2} dot={false} name="RUL MAE" />
                          </LineChart>
                        </ResponsiveContainer>
                      </div>
                    </div>
                  );
                })}
              </div>
            )}

        </main>
      </div>
    </div>
  );
}

export default App;
