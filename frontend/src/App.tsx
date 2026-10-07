import React, { useState, useEffect } from 'react';
import { api } from './api/client';
import { PieChart, Pie, Cell, Tooltip as RechartsTooltip, Legend as RechartsLegend } from 'recharts';
import Plot from 'react-plotly.js';

function App() {
  const [health, setHealth] = useState<string>('Checking...');
  const [summary, setSummary] = useState<any>(null);
  const [segments, setSegments] = useState<any[]>([]);
  const [scatterData, setScatterData] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);
  const [selectedK, setSelectedK] = useState<number>(5);

  useEffect(() => {
    api.checkHealth().then(() => setHealth('API Online')).catch(() => setHealth('API Offline'));
    api.getSummary().then(res => setSummary(res.data)).catch(console.error);
    fetchSegments();
  }, []);

  const fetchSegments = () => {
    api.getSegments().then(res => setSegments(res.data)).catch(() => setSegments([]));
  };

  const runClustering = async () => {
    setLoading(true);
    try {
      const features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)'];
      const res = await api.fitModel(features, selectedK);
      
      if (res.data.pca_points && res.data.labels) {
        const points = res.data.pca_points.map((pt: number[], i: number) => ({
          x: pt[0] || 0,
          y: pt[1] || 0,
          z: pt[2] || 0,
          cluster: res.data.labels[i]
        }));
        setScatterData(points);
      }
      
      await fetchSegments();
    } catch (e) {
      console.error(e);
      alert("Failed to run clustering.");
    }
    setLoading(false);
  };

  // Futuristic Neon Colors
  const COLORS = ['#00f3ff', '#ff00ff', '#00ff00', '#ffaa00', '#ff0055', '#a200ff', '#f3ff00', '#00ff88'];

  // Mapping cluster ID to Persona name for the 3D Plot Legend
  const getPersonaName = (clusterId: number) => {
    const seg = segments.find(s => s.cluster_id === clusterId);
    return seg ? seg.persona : `Cluster ${clusterId}`;
  };

  // Prepare data for Plotly 3D Scatter
  const plotData = Array.from(new Set(scatterData.map(d => d.cluster))).map((clusterId, i) => {
    const clusterPoints = scatterData.filter(d => d.cluster === clusterId);
    return {
      x: clusterPoints.map(p => p.x),
      y: clusterPoints.map(p => p.y),
      z: clusterPoints.map(p => p.z),
      type: 'scatter3d',
      mode: 'markers',
      marker: {
        size: 6,
        color: COLORS[i % COLORS.length],
        line: { color: '#ffffff', width: 0.5 },
        opacity: 0.8
      },
      name: getPersonaName(clusterId)
    };
  });

  return (
    <div className="min-h-screen bg-gray-900 text-white font-sans overflow-x-hidden selection:bg-pink-500 selection:text-white" 
         style={{ backgroundImage: 'radial-gradient(circle at 50% 0%, #1f2937 0%, #111827 50%, #030712 100%)' }}>
      
      {/* Premium Header with CSS 3D Effect */}
      <div className="pt-16 pb-8 px-8 max-w-7xl mx-auto text-center">
        <h1 className="text-6xl md:text-8xl font-black mb-4 tracking-tighter"
            style={{
              textShadow: `0 1px 0 #4b5563, 0 2px 0 #374151, 0 3px 0 #1f2937, 
                           0 4px 0 #111827, 0 5px 0 #000, 
                           0 6px 1px rgba(0,0,0,0.1), 0 0 5px rgba(0,0,0,0.1), 
                           0 1px 3px rgba(0,0,0,0.3), 0 3px 5px rgba(0,0,0,0.2), 
                           0 5px 10px rgba(0,0,0,0.25), 0 10px 10px rgba(0,0,0,0.2), 
                           0 20px 20px rgba(0,0,0,0.15)`
            }}>
          <span className="text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 via-purple-500 to-pink-500">
            NEXUS
          </span> SEGMENTS
        </h1>
        <p className="text-xl text-gray-400 font-light max-w-2xl mx-auto mb-8">
          Advanced 3D K-Means Clustering Intelligence System
        </p>
        
        {/* Control Panel (Glassmorphism) */}
        <div className="flex flex-col md:flex-row items-center justify-center gap-6 p-6 rounded-2xl backdrop-blur-xl bg-white/5 border border-white/10 shadow-2xl max-w-3xl mx-auto relative overflow-hidden">
          <div className="absolute inset-0 bg-gradient-to-r from-cyan-500/10 to-purple-500/10 pointer-events-none"></div>
          
          <div className="flex items-center gap-3 relative z-10">
            <div className={`w-3 h-3 rounded-full animate-pulse ${health.includes('Online') ? 'bg-green-400 shadow-[0_0_10px_#4ade80]' : 'bg-red-500'}`}></div>
            <span className="text-sm font-medium text-gray-300 uppercase tracking-widest">{health}</span>
          </div>
          
          <div className="w-px h-8 bg-white/20 hidden md:block"></div>
          
          <div className="flex items-center gap-4 relative z-10">
            <label className="font-medium text-gray-300">Target Clusters (K):</label>
            <input 
              type="number" min="2" max="10" 
              value={selectedK} 
              onChange={(e) => setSelectedK(parseInt(e.target.value))}
              className="bg-black/40 border border-white/20 text-white p-2 w-20 rounded-lg outline-none focus:ring-2 focus:ring-cyan-500 text-center text-lg font-bold"
            />
            <button 
              onClick={runClustering} 
              disabled={loading}
              className="relative group overflow-hidden bg-gradient-to-r from-cyan-500 to-purple-600 text-white font-bold py-3 px-8 rounded-lg shadow-[0_0_20px_rgba(6,182,212,0.4)] hover:shadow-[0_0_30px_rgba(168,85,247,0.6)] transition-all duration-300 disabled:opacity-50"
            >
              <div className="absolute inset-0 bg-white/20 translate-y-full group-hover:translate-y-0 transition-transform duration-300"></div>
              <span className="relative flex items-center gap-2">
                {loading ? (
                  <><svg className="animate-spin h-5 w-5 text-white" viewBox="0 0 24 24"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none"></circle><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg> PROCESSING...</>
                ) : '▶ INITIALIZE MODEL'}
              </span>
            </button>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-8 pb-16">
        {summary && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-12">
            <div className="p-6 rounded-2xl backdrop-blur-md bg-white/5 border border-white/10 flex items-center justify-between shadow-xl">
              <div>
                <h3 className="text-gray-400 uppercase tracking-widest text-xs font-bold mb-1">Total Profiles Analyzed</h3>
                <p className="text-4xl font-black text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-blue-500">{summary.shape[0]}</p>
              </div>
              <div className="p-4 bg-cyan-500/20 rounded-full border border-cyan-500/30"><span className="text-2xl">👥</span></div>
            </div>
            <div className="p-6 rounded-2xl backdrop-blur-md bg-white/5 border border-white/10 flex items-center justify-between shadow-xl">
              <div>
                <h3 className="text-gray-400 uppercase tracking-widest text-xs font-bold mb-1">Dimensionality</h3>
                <p className="text-4xl font-black text-transparent bg-clip-text bg-gradient-to-r from-purple-400 to-pink-500">{summary.shape[1]} Features</p>
              </div>
              <div className="p-4 bg-purple-500/20 rounded-full border border-purple-500/30"><span className="text-2xl">⚡</span></div>
            </div>
          </div>
        )}

        {scatterData.length > 0 && (
          <div className="backdrop-blur-xl bg-black/40 border border-white/10 rounded-3xl p-8 mb-12 shadow-2xl relative overflow-hidden">
            <div className="absolute top-0 right-0 w-96 h-96 bg-cyan-500/10 rounded-full blur-[100px] pointer-events-none"></div>
            <div className="absolute bottom-0 left-0 w-96 h-96 bg-purple-500/10 rounded-full blur-[100px] pointer-events-none"></div>
            
            <h2 className="text-3xl font-black mb-2 text-white relative z-10">3D Neural Segmentation Map</h2>
            <p className="text-gray-400 mb-8 relative z-10">Interactive geospatial clustering projection. Drag to rotate, scroll to zoom.</p>
            
            <div className="w-full flex justify-center bg-gray-900/50 rounded-2xl border border-white/5 p-4 relative z-10 shadow-[inset_0_0_20px_rgba(0,0,0,0.5)]">
              <Plot
                data={plotData as any}
                layout={{
                  width: window.innerWidth > 1000 ? 900 : window.innerWidth - 100,
                  height: 600,
                  paper_bgcolor: 'rgba(0,0,0,0)',
                  plot_bgcolor: 'rgba(0,0,0,0)',
                  margin: { l: 0, r: 0, b: 0, t: 0 },
                  font: { color: '#9ca3af' },
                  legend: { font: { color: '#ffffff', size: 14 }, bgcolor: 'rgba(0,0,0,0.5)', bordercolor: 'rgba(255,255,255,0.1)', borderwidth: 1 },
                  scene: {
                    xaxis: { title: 'Age (PCA)', gridcolor: '#374151', zerolinecolor: '#4b5563', showbackground: false },
                    yaxis: { title: 'Income (PCA)', gridcolor: '#374151', zerolinecolor: '#4b5563', showbackground: false },
                    zaxis: { title: 'Spending (PCA)', gridcolor: '#374151', zerolinecolor: '#4b5563', showbackground: false },
                    camera: { eye: { x: 1.5, y: 1.5, z: 1.2 } }
                  }
                }}
                config={{ displayModeBar: true, responsive: true }}
              />
            </div>
          </div>
        )}

        {segments.length > 0 && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            <div className="backdrop-blur-xl bg-white/5 border border-white/10 rounded-3xl p-8 shadow-2xl relative overflow-hidden">
              <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-64 h-64 bg-cyan-500/10 rounded-full blur-[80px] pointer-events-none"></div>
              <h2 className="text-2xl font-bold mb-6 text-white border-b border-white/10 pb-4 relative z-10">Segment Distribution</h2>
              <div className="flex justify-center relative z-10">
                <PieChart width={400} height={300}>
                  <Pie 
                    data={segments} 
                    dataKey="share_percent" 
                    nameKey="persona" 
                    cx="50%" cy="50%" 
                    innerRadius={80} outerRadius={120} 
                    stroke="none"
                    labelLine={false}
                  >
                    {segments.map((_, index) => (
                      <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} 
                            style={{ filter: `drop-shadow(0px 0px 8px ${COLORS[index % COLORS.length]}80)` }}/>
                    ))}
                  </Pie>
                  <RechartsTooltip 
                    contentStyle={{ backgroundColor: 'rgba(17,24,39,0.9)', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '8px', color: '#fff' }}
                    itemStyle={{ color: '#fff' }}
                  />
                  <RechartsLegend wrapperStyle={{ color: '#fff' }}/>
                </PieChart>
              </div>
            </div>
            
            <div className="backdrop-blur-xl bg-white/5 border border-white/10 rounded-3xl p-8 shadow-2xl flex flex-col">
              <h2 className="text-2xl font-bold mb-6 text-white border-b border-white/10 pb-4">Intelligence Profiles</h2>
              <div className="overflow-x-auto flex-1">
                <table className="w-full text-left border-collapse">
                  <thead>
                    <tr className="text-gray-400 uppercase text-xs tracking-wider border-b border-white/10">
                      <th className="p-4 font-bold">Persona Class</th>
                      <th className="p-4 font-bold text-right">Avg Income</th>
                      <th className="p-4 font-bold text-right">Avg Spending</th>
                      <th className="p-4 font-bold text-right">Avg Age</th>
                    </tr>
                  </thead>
                  <tbody>
                    {segments.map((seg, i) => (
                      <tr key={i} className="border-b border-white/5 hover:bg-white/5 transition-colors group">
                        <td className="p-4 font-bold text-white flex items-center gap-3">
                          <div className="w-4 h-4 rounded-full shadow-[0_0_10px_currentColor]" style={{ backgroundColor: COLORS[i % COLORS.length], color: COLORS[i % COLORS.length] }}></div>
                          {seg.persona}
                        </td>
                        <td className="p-4 text-right text-gray-300 group-hover:text-cyan-400 transition-colors">${seg.avg_Income?.toFixed(1) || 0}k</td>
                        <td className="p-4 text-right text-gray-300 group-hover:text-purple-400 transition-colors">{seg.avg_Spending?.toFixed(1) || 0}</td>
                        <td className="p-4 text-right text-gray-300 group-hover:text-pink-400 transition-colors">{seg.avg_Age?.toFixed(1) || 0} yrs</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
