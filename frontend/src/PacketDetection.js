import React, { useState } from "react";

function PacketDetection() {
  const [images, setImages] = useState([]);
  const [previews, setPreviews] = useState([]);
  const [results, setResults] = useState([]);
  const [message, setMessage] = useState("");

  const handleImageChange = (e) => {
    const files = Array.from(e.target.files);
    setImages(files);
    setPreviews(files.map((file) => URL.createObjectURL(file)));
    setResults([]);
    setMessage(""); // Clear previous message
  };

  const handleUpload = async () => {
    const formData = new FormData();
    images.forEach((image) => formData.append("images", image));

    const res = await fetch("http://localhost:5000/packet-upload-multiple", {
      method: "POST",
      body: formData,
    });

    const data = await res.json();

    if (data.results) {
      setResults(data.results);
      setMessage("✅ Item have been added to the inventory.");
    } else {
      setMessage("⚠️ No items were added. Something went wrong.");
    }
  };

  return (
    <div className="container mt-4">
      <h2 className="mb-4 text-center">📦 Packet Detection</h2>

      <div className="mb-3">
        <input
          type="file"
          multiple
          accept="image/*"
          onChange={handleImageChange}
          className="form-control"
        />
      </div>

      {previews.length > 0 && (
        <div className="mb-4">
          <h5>Selected Images:</h5>
          <div className="d-flex flex-wrap gap-3">
            {previews.map((src, i) => (
              <img
                key={i}
                src={src}
                alt={`Preview ${i}`}
                height={150}
                className="rounded shadow"
              />
            ))}
          </div>

          <div className="mt-3">
            <button className="btn btn-primary" onClick={handleUpload}>
              🚀 Upload & Detect
            </button>

            {message && (
              <div className="alert alert-success mt-3" role="alert">
                {message}
              </div>
            )}
          </div>
        </div>
      )}

      {results.length > 0 && (
        <div className="mt-5">
          <h4 className="text-success">Detection Results:</h4>
          <div className="row">
            {results.map((res, i) => {
              const parsed = safeJSONParse(res.extracted_info);
              return (
                <div key={i} className="col-md-6 mb-4">
                  <div className="card shadow">
                    <img
                      src={res.result_url}
                      alt={`Result ${i}`}
                      className="card-img-top"
                      style={{ height: "200px", objectFit: "contain" }}
                    />
                    <div className="card-body">
                      <h6>Extracted Info:</h6>
                      <pre style={{ whiteSpace: "pre-wrap", fontFamily: "monospace" }}>
                        {parsed ? (
                          <>
                            Ingredient: {parsed.ingredient || "null"}{"\n"}
                            Total Weight: {parsed.total_weight || "null"}{"\n"}
                            Expiry Date: {parsed.expiry_date || "null"}
                          </>
                        ) : (
                          res.extracted_info
                        )}
                      </pre>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}

function safeJSONParse(str) {
  try {
    return JSON.parse(str);
  } catch (e) {
    return null;
  }
}

export default PacketDetection;
