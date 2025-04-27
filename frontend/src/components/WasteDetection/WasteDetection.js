// import React, { useState } from "react";
// import axios from "axios";

// function WasteDetection() {
//   const [selectedImage, setSelectedImage] = useState(null);
//   const [predictions, setPredictions] = useState([]);
//   const [annotatedImage, setAnnotatedImage] = useState(null);
//   const [loading, setLoading] = useState(false);

//   const handleImageChange = (e) => {
//     setSelectedImage(e.target.files[0]);
//     setPredictions([]);
//     setAnnotatedImage(null);
//   };

//   const handleSubmit = async () => {
//     if (!selectedImage) return alert("Please select an image first.");

//     const formData = new FormData();
//     formData.append("image", selectedImage);

//     setLoading(true);
//     try {
//       const res = await axios.post("http://localhost:5000/api/waste-detection", formData);
//       setPredictions(res.data.predictions);
//       setAnnotatedImage(`data:image/png;base64,${res.data.annotated_image}`);
//     } catch (error) {
//       console.error("Error:", error);
//       alert("Error detecting waste. Please check backend.");
//     } finally {
//       setLoading(false);
//     }
//   };

//   return (
//     <div style={{ padding: "20px" }}>
//       <h2>Waste Detection</h2>
//       <input type="file" accept="image/*" onChange={handleImageChange} />
//       <button onClick={handleSubmit} disabled={loading} style={{ marginLeft: "10px" }}>
//         {loading ? "Detecting..." : "Submit"}
//       </button>

//       {annotatedImage && (
//         <div style={{ marginTop: "20px" }}>
//           <h4>Annotated Image:</h4>
//           <img src={annotatedImage} alt="Annotated Result" style={{ maxWidth: "100%", border: "1px solid #ccc" }} />
//         </div>
//       )}

//       {predictions.length > 0 && (
//         <div style={{ marginTop: "20px" }}>
//           <h4>Predictions:</h4>
//           <ul>
//             {predictions.map((p, index) => (
//               <li key={index}>
//                 <strong>{p.class}</strong> – Confidence: {(p.confidence * 100).toFixed(2)}%
//               </li>
//             ))}
//           </ul>
//         </div>
//       )}
//     </div>
//   );
// }

// export default WasteDetection;
import React, { useState } from "react";
import axios from "axios";

function WasteDetection() {
  const [selectedImage, setSelectedImage] = useState(null);
  const [predictions, setPredictions] = useState([]);
  const [annotatedImage, setAnnotatedImage] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleImageChange = (e) => {
    setSelectedImage(e.target.files[0]);
    setPredictions([]);
    setAnnotatedImage(null);
  };

  const handleSubmit = async () => {
    if (!selectedImage) return alert("Please select an image first.");

    const formData = new FormData();
    formData.append("image", selectedImage);

    setLoading(true);
    try {
      const res = await axios.post("http://localhost:5000/api/waste-detection", formData);
      setPredictions(res.data.predictions);
      setAnnotatedImage(`data:image/png;base64,${res.data.annotated_image}`);
    } catch (error) {
      console.error("Error:", error);
      alert("Error detecting waste. Please check backend.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container mt-5">
      <h2 className="mb-4 text-center">🗑️ Waste Detection</h2>

      <div className="mb-3">
        <input
          type="file"
          accept="image/*"
          onChange={handleImageChange}
          className="form-control"
        />
      </div>

      <div className="mb-4">
        <button
          onClick={handleSubmit}
          disabled={loading}
          className="btn btn-primary"
        >
          {loading ? "🔍 Detecting..." : "🚀 Submit"}
        </button>
      </div>

      {annotatedImage && (
        <div className="mb-5">
          <h4 className="text-success">Annotated Image:</h4>
          <div className="text-center">
            <img
              src={annotatedImage}
              alt="Annotated Result"
              className="img-fluid rounded shadow border"
              style={{ maxHeight: "500px" }}
            />
          </div>
        </div>
      )}

      {predictions.length > 0 && (
        <div className="mb-4">
          <h4 className="text-info">Predictions:</h4>
          <ul className="list-group">
            {predictions.map((p, index) => (
              <li key={index} className="list-group-item d-flex justify-content-between align-items-center">
                <span><strong>{p.class}</strong></span>
                <span className="badge bg-secondary">
                  Confidence: {(p.confidence * 100).toFixed(2)}%
                </span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

export default WasteDetection;
