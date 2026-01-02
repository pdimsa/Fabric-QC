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
                        {/* Header */}
                        <div className="result-header">
                            <h3>Analysis Result</h3>
                            <span className={`status-badge ${result.prediction.toLowerCase()}`}>
                                {result.prediction}
                            </span>
                        </div>

                        {/* Defect Type / Classification */}
                        <div className="defect-type-section">
                            <div className="defect-type-label">Predicted Class</div>
                            <div className={`defect-type-value ${result.is_defective ? 'defective' : 'normal'}`}>
                                {result.defect_type}
                            </div>
                        </div>

                        {/* Confidence Bar */}
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

                        {/* Per-Class Probabilities */}
                        {result.class_probabilities && (
                            <div className="classification-section">
                                <h4>Class Probabilities</h4>
                                {result.class_probabilities.map((item, index) => {
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
                )}
            </main>

            <footer className="footer">
                <p>Powered by ResNet50 &amp; DCGAN • FabricGuard AI v2.0</p>
            </footer>
        </div>
    );
}

export default App;
