import { useEffect, useRef, useState } from "react";
import "./App.css";

export default function App() {
  const videoRef = useRef(null);
  const captureCanvasRef = useRef(null);
  const displayImageRef = useRef(null);

 
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [streaming, setStreaming] = useState(false);
  const [sourceType, setSourceType] = useState(null); // "upload" or "webcam"

  useEffect(() => {
    return () => {
      stopCamera();
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    };
  }, []);

  useEffect(() => {
    const handleKeyDown = async (event) => {
      if (event.code === "Space") {
        const tag = document.activeElement?.tagName;
        if (tag === "INPUT" || tag === "TEXTAREA" || tag === "BUTTON") return;
        event.preventDefault();
        if (streaming && !loading) {
          await captureFromWebcam();
        }
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [streaming, loading]);

  const stopCamera = () => {
    const video = videoRef.current;
    if (video && video.srcObject) {
      const tracks = video.srcObject.getTracks();
      tracks.forEach((track) => track.stop());
      video.srcObject = null;
    }
    setStreaming(false);
    setCameraReady(false);
  };

  const [cameraReady, setCameraReady] = useState(false);

  const disposalGuidance = {
    cardboard: {
      title: "Cardboard",
      message:
        "Flatten clean cardboard before recycling. Keep it dry, and throw it away if it is heavily soaked with food or grease.",
    },
    glass: {
      title: "Glass",
      message:
        "Rinse glass containers before recycling. Remove obvious non-glass parts when possible. Broken glass may need to go in the trash depending on local rules.",
    },
    metal: {
      title: "Metal",
      message:
        "Rinse metal cans and containers before recycling. Empty food cans and many aluminum containers are commonly accepted.",
    },
    paper: {
      title: "Paper",
      message:
        "Recycle clean, dry paper. Do not recycle paper that is heavily contaminated with food, oil, or liquids.",
    },
    plastic: {
      title: "Plastic",
      message:
        "Rinse plastic containers before recycling. Check local rules because not all plastic types are accepted in curbside recycling.",
    },
    trash: {
      title: "Trash",
      message:
        "This item is likely general waste. Place it in the trash unless your local facility has a special drop-off or recovery program for it.",
    },
    other: {
      title: "Other",
      message:
        "This item may need special handling. Check your local recycling or waste program to see whether it belongs in recycling, trash, or a specialty drop-off stream.",
    },
  };

  const startCamera = async () => {
    try {
      setError("");
      setResult(null);
      setCameraReady(false);
      setSourceType("webcam");

      // Let React render the video element first
      await new Promise((resolve) => setTimeout(resolve, 0));

      const stream = await navigator.mediaDevices.getUserMedia({
        video: true,
        audio: false,
      });

      const video = videoRef.current;
      if (!video) {
        setError("Video element not ready.");
        return;
      }

      video.srcObject = stream;

      await new Promise((resolve, reject) => {
        video.onloadedmetadata = async () => {
          try {
            await video.play();
            resolve();
          } catch (err) {
            reject(err);
          }
        };
      });

      setStreaming(true);
      setCameraReady(true);
    } catch (err) {
      console.error(err);
      setError("Could not access webcam.");
      setStreaming(false);
      setCameraReady(false);
    }
  };

  const runDetectionFromBlob = async (blob, previewObjectUrl, source) => {
    setLoading(true);
    setError("");
    setResult(null);

    try {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
      setPreviewUrl(previewObjectUrl);
      setSourceType(source);

      const formData = new FormData();
      formData.append("file", blob, "image.jpg");

      const response = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (!response.ok || data.error) {
        throw new Error(data.error || "Prediction failed.");
      }

      setResult(data);
    } catch (err) {
      console.error(err);
      setError(err.message || "Something went wrong.");
    } finally {
      setLoading(false);
    }
  };

  const handleFileChange = async (event) => {
    const file = event.target.files?.[0];
    if (!file) return;

    stopCamera();
    setSelectedFile(file);
    setResult(null);
    setError("");

    const objectUrl = URL.createObjectURL(file);
    await runDetectionFromBlob(file, objectUrl, "upload");
  };

  const captureFromWebcam = async () => {
    if (!videoRef.current || !captureCanvasRef.current) return;
    if (!cameraReady) {
      setError("Webcam is still loading.");
      return;
    }

    const video = videoRef.current;
    const canvas = captureCanvasRef.current;

    if (video.videoWidth === 0 || video.videoHeight === 0) {
      setError("Webcam is not ready yet.");
      return;
    }

    const targetWidth = 640;
    const scale = targetWidth / video.videoWidth;
    const targetHeight = Math.round(video.videoHeight * scale);

    canvas.width = targetWidth;
    canvas.height = targetHeight;

    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0, targetWidth, targetHeight);

    const blob = await new Promise((resolve) =>
      canvas.toBlob(resolve, "image/jpeg", 0.9)
    );

    if (!blob) {
      setError("Failed to capture frame.");
      return;
    }

    const objectUrl = URL.createObjectURL(blob);
    await runDetectionFromBlob(blob, objectUrl, "webcam");
  };

  const detectedClasses = result
    ? [...new Set(result.detections.map((det) => det.class.toLowerCase()))]
    : [];

  const detectedGuidance = detectedClasses
    .map((cls) => disposalGuidance[cls])
    .filter(Boolean);

  return (
    <div className="app-shell">
      <div className="app-container">
        <div className="app-header">
          <h1 className="app-title">Smart Cycle</h1>
          <p className="app-subtitle">
            Upload an image or capture a frame from your webcam for detection.
          </p>
        </div>

        <div className="control-bar">
          <input
            className="file-input"
            type="file"
            accept="image/*"
            onChange={handleFileChange}
          />

          <button
            className="button button-primary"
            onClick={startCamera}
            disabled={streaming}
          >
            Start Webcam
          </button>

          <button
            className="button button-secondary"
            onClick={stopCamera}
            disabled={!streaming}
          >
            Stop Webcam
          </button>

          <button
            className="button button-primary"
            onClick={captureFromWebcam}
            disabled={!streaming || !cameraReady || loading}
          >
            {loading ? "Detecting..." : "Capture + Detect"}
          </button>
        </div>

        <p className="status-text">
          {streaming
            ? cameraReady
              ? "Webcam ready. Press Space or click Capture + Detect."
              : "Starting webcam..."
            : "Upload an image or start the webcam."}
        </p>

        {error && <div className="error-banner">{error}</div>}

        <div className="main-grid">
          <div className="card">
            <h2 className="card-title">
              {sourceType === "webcam" && streaming
                ? "Live Webcam"
                : previewUrl
                ? sourceType === "webcam"
                  ? "Captured Frame"
                  : "Uploaded Image"
                : "Preview"}
            </h2>

            {sourceType === "webcam" && (
              <div className="media-frame">
                <video
                  ref={videoRef}
                  autoPlay
                  playsInline
                  muted
                  className="webcam-video"
                />
              </div>
            )}

            {previewUrl && (
              <div className="media-frame" style={{ marginTop: streaming ? "16px" : "0" }}>
                <img
                  ref={displayImageRef}
                  src={previewUrl}
                  alt="Detection preview"
                  className="preview-image"
                />

                {result &&
                  result.detections.map((det, index) => {
                    const [x1, y1, x2, y2] = det.bbox;
                    const width = result.image_size.width;
                    const height = result.image_size.height;

                    return (
                      <div
                        key={index}
                        className="detection-box"
                        style={{
                          left: `${(x1 / width) * 100}%`,
                          top: `${(y1 / height) * 100}%`,
                          width: `${((x2 - x1) / width) * 100}%`,
                          height: `${((y2 - y1) / height) * 100}%`,
                        }}
                      >
                        <div className="detection-label">
                          {det.class} ({det.confidence})
                        </div>
                      </div>
                    );
                  })}
              </div>
            )}

            {!streaming && !previewUrl && (
              <p className="helper-text">
                No media selected yet. Start the webcam or upload an image to begin.
              </p>
            )}
          </div>

          <div className="card">
            <h2 className="card-title">Detections</h2>

            {result ? (
              result.detections.length === 0 ? (
                <p className="empty-text">No objects detected.</p>
              ) : (
                <>
                  <ul className="result-list">
                    {result.detections.map((det, index) => (
                      <li key={index} className="result-item">
                        <div className="result-class">{det.class}</div>
                        <div>Confidence: {det.confidence}</div>
                        <div>
                          BBox: [{det.bbox.map((v) => Math.round(v)).join(", ")}]
                        </div>
                      </li>
                    ))}
                  </ul>

                  {detectedGuidance.length > 0 && (
                    <div style={{ marginTop: "1.5rem" }}>
                      <h3
                        style={{
                          margin: "0 0 0.75rem 0",
                          fontSize: "1rem",
                          color: "#111827",
                        }}
                      >
                        Disposal Guidance
                      </h3>

                      {detectedGuidance.map((item, index) => (
                        <div
                          key={index}
                          style={{
                            marginBottom: "0.75rem",
                            padding: "0.75rem",
                            background: "#f9fafb",
                            border: "1px solid #e5e7eb",
                            borderRadius: "10px",
                          }}
                        >
                          <div style={{ fontWeight: 700, marginBottom: "0.25rem" }}>
                            {item.title}
                          </div>
                          <div>{item.message}</div>
                        </div>
                      ))}

                      <p
                        style={{
                          fontSize: "0.9rem",
                          color: "#6b7280",
                          marginTop: "0.75rem",
                          marginBottom: 0,
                        }}
                      >
                        Disposal rules vary by location. Check your local recycling
                        program for final guidance.
                      </p>
                    </div>
                  )}
                </>
              )
            ) : (
              <p className="empty-text">
                Detection results will appear here after you upload an image or capture a frame.
              </p>
            )}
          </div>
        </div>

        <canvas ref={captureCanvasRef} style={{ display: "none" }} />
      </div>
    </div>
  );
}