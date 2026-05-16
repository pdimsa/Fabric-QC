import React, { useState, useRef, useCallback, useEffect } from 'react';
import axios from 'axios';
import './App.css';

const API_BASE = 'https://poojathmidimansa-fabricqcbe.hf.space';

function App() {
    // ── Mode Toggle ──────────────────────────────────────────────────────
    const [mode, setMode] = useState('upload'); // 'upload' | 'live'

    // ── Upload Mode State ────────────────────────────────────────────────
    const [selectedFile, setSelectedFile] = useState(null);
    const [preview, setPreview] = useState(null);
    const [result, setResult] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [dragActive, setDragActive] = useState(false);
    const fileInputRef = useRef(null);

    // ── Live Mode State ──────────────────────────────────────────────────
    const videoRef = useRef(null);
    const canvasRef = useRef(null);
    const streamRef = useRef(null);
    const intervalRef = useRef(null);
    const [cameraActive, setCameraActive] = useState(false);
    const [liveResult, setLiveResult] = useState(null);
    const [liveError, setLiveError] = useState(null);
    const [analyzing, setAnalyzing] = useState(false);
    const [fps, setFps] = useState(0);
    const [captureInterval, setCaptureInterval] = useState(1500); // ms between frames

    // ── Upload Handlers ──────────────────────────────────────────────────
    const handleFileChange = (file) => {
        if (file && file.type.startsWith('image/')) {
            setSelectedFile(file);
            setPreview(URL.createObjectURL(file));
            setResult(null);
            setError(null);
        } else {
            setError('Please upload a valid image file.');
        }
    };

    const onDrag = (e) => {
        e.preventDefault();
        e.stopPropagation();
        if (e.type === 'dragenter' || e.type === 'dragover') {
            setDragActive(true);
        } else if (e.type === 'dragleave') {
            setDragActive(false);
        }
    };

    const onDrop = (e) => {
        e.preventDefault();
        e.stopPropagation();
        setDragActive(false);
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            handleFileChange(e.dataTransfer.files[0]);
        }
    };

    const handleUpload = async () => {
        if (!selectedFile) return;

        setLoading(true);
        setError(null);

        const formData = new FormData();
        formData.append('file', selectedFile);

        try {
            const response = await axios.post(`${API_BASE}/predict`, formData, {
                headers: {
                    'Content-Type': 'multipart/form-data',
                },
            });

            if (response.data.error) {
                setError(response.data.error);
            } else {
                setResult(response.data);
            }
        } catch (err) {
            console.error(err);
            setError('Analysis failed. Please check your connection to the server.');
        } finally {
            setLoading(false);
        }
    };

    const triggerFileInput = () => {
        fileInputRef.current.click();
    };

    const resetSelection = () => {
        setSelectedFile(null);
        setPreview(null);
        setResult(null);
        setError(null);
    };

    // ── Live Camera Handlers ─────────────────────────────────────────────
    const startCamera = useCallback(async () => {
        try {
            setLiveError(null);
            const stream = await navigator.mediaDevices.getUserMedia({
                video: { facingMode: 'environment', width: { ideal: 640 }, height: { ideal: 480 } }
            });
            streamRef.current = stream;
            if (videoRef.current) {
                videoRef.current.srcObject = stream;
            }
            setCameraActive(true);
        } catch (err) {
            console.error('Camera error:', err);
            setLiveError('Unable to access camera. Please check permissions.');
        }
    }, []);

    const stopCamera = useCallback(() => {
        if (intervalRef.current) {
            clearInterval(intervalRef.current);
            intervalRef.current = null;
        }
        if (streamRef.current) {
            streamRef.current.getTracks().forEach(track => track.stop());
            streamRef.current = null;
        }
        if (videoRef.current) {
            videoRef.current.srcObject = null;
        }
        setCameraActive(false);
        setAnalyzing(false);
        setLiveResult(null);
        setFps(0);
    }, []);

    const captureAndAnalyze = useCallback(async () => {
        if (!videoRef.current || !canvasRef.current) return;

        const video = videoRef.current;
        const canvas = canvasRef.current;
        canvas.width = video.videoWidth || 640;
        canvas.height = video.videoHeight || 480;

        const ctx = canvas.getContext('2d');
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

        const frameDataUrl = canvas.toDataURL('image/jpeg', 0.8);

        try {
            const startTime = performance.now();
            const response = await axios.post(`${API_BASE}/predict/frame`, {
                frame: frameDataUrl
            });
            const elapsed = performance.now() - startTime;
            setFps(Math.round(1000 / elapsed));

            if (response.data.error) {
                setLiveError(response.data.error);
            } else {
                setLiveResult(response.data);
                setLiveError(null);
            }
        } catch (err) {
            console.error('Frame analysis error:', err);
            // Don't set error for transient failures during live mode
        }
    }, []);

    const startAnalysis = useCallback(() => {
        if (!cameraActive) return;
        setAnalyzing(true);
        setLiveResult(null);

        // Initial capture
        captureAndAnalyze();

        // Periodic captures
        intervalRef.current = setInterval(() => {
            captureAndAnalyze();
        }, captureInterval);
    }, [cameraActive, captureAndAnalyze, captureInterval]);

    const stopAnalysis = useCallback(() => {
        if (intervalRef.current) {
            clearInterval(intervalRef.current);
            intervalRef.current = null;
        }
        setAnalyzing(false);
        setFps(0);
    }, []);

    // Clean up on mode switch or unmount
    useEffect(() => {
        if (mode === 'upload') {
            stopCamera();
        }
        return () => {
            stopCamera();
        };
    }, [mode, stopCamera]);

    // Restart interval when capture speed changes
    useEffect(() => {
        if (analyzing) {
            if (intervalRef.current) clearInterval(intervalRef.current);
            intervalRef.current = setInterval(() => {
                captureAndAnalyze();
            }, captureInterval);
        }
    }, [captureInterval, analyzing, captureAndAnalyze]);

    // ── Results Component (shared) ───────────────────────────────────────
    const ResultDisplay = ({ data, compact = false }) => {
        if (!data) return null;
        return (
            <div className={`card result-card ${data.prediction.toLowerCase()} ${compact ? 'compact' : ''}`}>
                {/* Header */}
                <div className="result-header">
                    <h3>{compact ? 'Live Result' : 'Analysis Result'}</h3>
                    <span className={`status-badge ${data.prediction.toLowerCase()}`}>
                        {data.prediction}
                    </span>
                </div>

                {/* Defect Type / Classification */}
                <div className="defect-type-section">
                    <div className="defect-type-label">Predicted Class</div>
                    <div className={`defect-type-value ${data.is_defective ? 'defective' : 'normal'}`}>
                        {data.defect_type}
                    </div>
                </div>

                {/* Confidence Bar */}
                <div className="metric">
                    <span className="metric-label">Confidence</span>
                    <div className="progress-bar-bg">
                        <div
                            className="progress-bar-fill"
                            style={{ width: data.confidence }}
                        ></div>
                    </div>
                    <span className="metric-value">{data.confidence}</span>
                </div>

                {/* Per-Class Probabilities */}
                {data.class_probabilities && !compact && (
                    <div className="classification-section">
                        <h4>Class Probabilities</h4>
                        {data.class_probabilities.map((item, index) => {
                            const isTop = index === 0;
                            const isDefectFree = item.class_name === 'defect free';
                            const pctWidth = `${(item.probability * 100).toFixed(1)}%`;
                            const pctLabel = `${(item.probability * 100).toFixed(1)}%`;

                            return (
                                <div className="class-prob-row" key={item.class_name}>
                                    <span className={`class-name ${isTop ? 'top-class' : ''}`}>
                                        {item.class_name}
                                    </span>
                                    <div className="class-bar-bg">
                                        <div
                                            className={`class-bar-fill ${isTop ? 'top-bar' : ''} ${isDefectFree ? 'defect-free-bar' : ''}`}
                                            style={{ width: pctWidth }}
                                        ></div>
                                    </div>
                                    <span className={`class-prob-value ${isTop ? 'top-value' : ''}`}>
                                        {pctLabel}
                                    </span>
                                </div>
                            );
                        })}
                    </div>
                )}
            </div>
        );
    };

    return (
        <div className="app-container">
            <nav className="navbar">
                <div className="logo">
                    <span className="logo-icon">🔍</span>
                    <h1>FabricGuard AI</h1>
                </div>

                {/* Mode Toggle */}
                <div className="mode-toggle" id="mode-toggle">
                    <button
                        className={`mode-btn ${mode === 'upload' ? 'active' : ''}`}
                        onClick={() => setMode('upload')}
                        id="mode-upload-btn"
                    >
                        <span className="mode-icon">📁</span>
                        Upload
                    </button>
                    <button
                        className={`mode-btn ${mode === 'live' ? 'active' : ''}`}
                        onClick={() => setMode('live')}
                        id="mode-live-btn"
                    >
                        <span className="mode-icon">📹</span>
                        Live Cam
                    </button>
                    <div className={`mode-slider ${mode === 'live' ? 'right' : 'left'}`}></div>
                </div>
            </nav>

            <main className="main-content">
                {/* ═════════════════════  UPLOAD MODE  ═════════════════════ */}
                {mode === 'upload' && (
                    <div className="card upload-card" id="upload-panel">
                        <h2>Defect Detection</h2>
                        <p className="subtitle">Upload a fabric sample to analyze for defects using our GAN-augmented ResNet50 model.</p>

                        <div
                            className={`drop-zone ${dragActive ? 'active' : ''} ${preview ? 'has-preview' : ''}`}
                            onDragEnter={onDrag}
                            onDragLeave={onDrag}
                            onDragOver={onDrag}
                            onDrop={onDrop}
                            onClick={!preview ? triggerFileInput : undefined}
                        >
                            <input
                                ref={fileInputRef}
                                type="file"
                                accept="image/*"
                                onChange={(e) => handleFileChange(e.target.files[0])}
                                className="hidden-input"
                                id="file-input"
                            />

                            {!preview ? (
                                <div className="drop-zone-content">
                                    <div className="upload-icon">☁️</div>
                                    <p><strong>Click to upload</strong> or drag and drop</p>
                                    <p className="file-hint">SVG, PNG, JPG or GIF</p>
                                </div>
                            ) : (
                                <div className="preview-container">
                                    <img src={preview} alt="Selected fabric" className="preview-image" />
                                    <button className="remove-btn" onClick={(e) => { e.stopPropagation(); resetSelection(); }}>
                                        ×
                                    </button>
                                </div>
                            )}

                            {dragActive && <div className="drag-overlay">Drop image here</div>}
                        </div>

                        {error && <div className="error-message">⚠️ {error}</div>}

                        <button
                            onClick={handleUpload}
                            disabled={!selectedFile || loading}
                            className={`analyze-btn ${loading ? 'loading' : ''}`}
                            id="analyze-btn"
                        >
                            {loading ? <span className="spinner"></span> : 'Analyze Fabric'}
                        </button>
                    </div>
                )}

                {mode === 'upload' && result && <ResultDisplay data={result} />}

                {/* ═════════════════════  LIVE MODE  ══════════════════════ */}
                {mode === 'live' && (
                    <>
                        <div className="card live-card" id="live-panel">
                            <h2>Live Camera Detection</h2>
                            <p className="subtitle">Point your camera at fabric samples for real-time defect analysis.</p>

                            {/* Camera Viewport */}
                            <div className={`camera-viewport ${cameraActive ? 'active' : ''}`}>
                                <video
                                    ref={videoRef}
                                    autoPlay
                                    playsInline
                                    muted
                                    className="camera-feed"
                                />
                                <canvas ref={canvasRef} className="hidden-canvas" />

                                {!cameraActive && (
                                    <div className="camera-placeholder" onClick={startCamera}>
                                        <div className="camera-placeholder-icon">📷</div>
                                        <p><strong>Click to start camera</strong></p>
                                        <p className="file-hint">Requires camera permission</p>
                                    </div>
                                )}

                                {/* Scanning overlay */}
                                {analyzing && (
                                    <div className="scan-overlay">
                                        <div className="scan-line"></div>
                                        <div className="scan-corner tl"></div>
                                        <div className="scan-corner tr"></div>
                                        <div className="scan-corner bl"></div>
                                        <div className="scan-corner br"></div>
                                    </div>
                                )}

                                {/* Live result badge overlay */}
                                {analyzing && liveResult && (
                                    <div className={`live-badge ${liveResult.prediction.toLowerCase()}`}>
                                        <span className="live-badge-dot"></span>
                                        <span className="live-badge-text">
                                            {liveResult.defect_type} — {liveResult.confidence}
                                        </span>
                                    </div>
                                )}

                                {/* FPS counter */}
                                {analyzing && fps > 0 && (
                                    <div className="fps-counter">
                                        ~{fps} inf/s
                                    </div>
                                )}
                            </div>

                            {liveError && <div className="error-message">⚠️ {liveError}</div>}

                            {/* Camera Controls */}
                            <div className="live-controls">
                                {!cameraActive ? (
                                    <button className="analyze-btn" onClick={startCamera} id="start-camera-btn">
                                        <span className="btn-icon">📷</span> Start Camera
                                    </button>
                                ) : (
                                    <>
                                        <div className="controls-row">
                                            {!analyzing ? (
                                                <button className="analyze-btn live-start" onClick={startAnalysis} id="start-analysis-btn">
                                                    <span className="btn-icon">⚡</span> Start Analysis
                                                </button>
                                            ) : (
                                                <button className="analyze-btn live-stop" onClick={stopAnalysis} id="stop-analysis-btn">
                                                    <span className="btn-icon">⏸️</span> Pause Analysis
                                                </button>
                                            )}
                                            <button className="analyze-btn secondary-btn" onClick={stopCamera} id="stop-camera-btn">
                                                <span className="btn-icon">⏹️</span> Stop
                                            </button>
                                        </div>

                                        {/* Speed Slider */}
                                        <div className="speed-control">
                                            <label className="speed-label">
                                                Analysis Speed
                                                <span className="speed-value">{(1000 / captureInterval).toFixed(1)} fps</span>
                                            </label>
                                            <input
                                                type="range"
                                                min="300"
                                                max="5000"
                                                step="100"
                                                value={captureInterval}
                                                onChange={(e) => setCaptureInterval(Number(e.target.value))}
                                                className="speed-slider"
                                                id="speed-slider"
                                            />
                                            <div className="speed-labels">
                                                <span>Fast</span>
                                                <span>Slow</span>
                                            </div>
                                        </div>
                                    </>
                                )}
                            </div>
                        </div>

                        {/* Full result card below */}
                        {liveResult && <ResultDisplay data={liveResult} />}
                    </>
                )}
            </main>

            <footer className="footer">
                <p>Powered by ResNet50 &amp; DCGAN • FabricGuard AI v2.0</p>
            </footer>
        </div>
    );
}

export default App;
