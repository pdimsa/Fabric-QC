import React, { useState, useRef } from 'react';
import axios from 'axios';
import './App.css';

function App() {
    const [selectedFile, setSelectedFile] = useState(null);
    const [preview, setPreview] = useState(null);
    const [result, setResult] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [dragActive, setDragActive] = useState(false);

    const fileInputRef = useRef(null);

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
            const response = await axios.post('http://localhost:8000/predict', formData, {
                headers: {
                    'Content-Type': 'multipart/form-data',
                },
            });
            setResult(response.data);
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

    return (
        <div className="app-container">
            <nav className="navbar">
                <div className="logo">
                    <span className="logo-icon">🔍</span>
                    <h1>FabricGuard AI</h1>
                </div>
            </nav>

            <main className="main-content">
                <div className="card upload-card">
                    <h2>Defect Detection</h2>
                    <p className="subtitle">Upload a fabric sample to analyze for defects.</p>

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
                    >
                        {loading ? <span className="spinner"></span> : 'Analyze Fabric'}
                    </button>
                </div>

                {result && (
                    <div className={`card result-card ${result.prediction.toLowerCase()}`}>
                        <div className="result-header">
                            <h3>Analysis Result</h3>
                            <span className={`status-badge ${result.prediction.toLowerCase()}`}>
                                {result.prediction}
                            </span>
                        </div>

                        <div className="result-details">
                            <div className="metric">
                                <span className="metric-label">Confidence</span>
                                <div className="progress-bar-bg">
                                    <div
                                        className="progress-bar-fill"
                                        style={{ width: result.confidence }}
                                    ></div>
                                </div>
                                <span className="metric-value">{result.confidence}</span>
                            </div>

                            <div className="metric-row">
                                <span className="metric-label">Probability Score</span>
                                <span className="metric-value monospace">{result.probability.toFixed(4)}</span>
                            </div>
                        </div>
                    </div>
                )}
            </main>

            <footer className="footer">
                <p>Powered by ResNet18 & DCGAN • FabricGuard AI v1.0</p>
            </footer>
        </div>
    );
}

export default App;
