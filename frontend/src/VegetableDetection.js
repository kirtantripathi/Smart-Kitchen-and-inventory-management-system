import React, { useState } from "react";
import 'bootstrap/dist/css/bootstrap.min.css';

function VegetableDetection() {
  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [resultUrl, setResultUrl] = useState(null);
  const [summary, setSummary] = useState("");

  const handleImageChange = (e) => {
    const file = e.target.files[0];
    setImage(file);
    setPreview(URL.createObjectURL(file));
    setResultUrl(null);
    setSummary("");
  };

  const handleUpload = async () => {
    if (!image) return;

    const formData = new FormData();
    formData.append("image", image);

    try {
      const res = await fetch("http://localhost:5000/upload", {
        method: "POST",
        body: formData,
      });

      const data = await res.json();

      if (data.result_url) {
        setResultUrl(data.result_url);
      }
      if (data.summary) {
        setSummary(data.summary);
      }
    } catch (error) {
      console.error("Upload failed:", error);
    }
  };

  return (
    <div className="container py-5">
      <div className="text-center mb-4">
        <h2 className="fw-bold text-success">🥕Fruits & Vegetable Health Detection</h2>
        <p className="text-secondary">Upload an image to detect using AI.</p>
      </div>

      <div className="row justify-content-center">
        <div className="col-md-6">
          <input type="file" className="form-control mb-3" onChange={handleImageChange} />
        </div>
      </div>

      {preview && (
        <div className="text-center">
          <img
            src={preview}
            alt="Preview"
            className="img-thumbnail mb-3"
            style={{ maxHeight: "300px" }}
          />
          <br />
          <button className="btn btn-success px-4" onClick={handleUpload}>
            🔍 Detect
          </button>
        </div>
      )}

      {resultUrl && (
        <div className="text-center mt-5">
          <h4 className="text-primary mb-3">🍅 Detection Result</h4>
          <img
            src={resultUrl}
            alt="Result"
            className="img-fluid rounded"
            style={{ maxHeight: "300px" }}
          />
          {summary && (
            <div className="mt-3">
              <h5 className="text-dark">🧾 Summary</h5>
              <p className="text-muted">{summary}</p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default VegetableDetection;
