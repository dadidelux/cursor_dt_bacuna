import React, { useState } from "react";
import axios from "axios";

const API_BASE_URL = "http://67.217.58.19:8000";

function App() {
  const [file, setFile] = useState(null);
  const [predictionType, setPredictionType] = useState("disease");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) return alert("Please select an image file.");

    const formData = new FormData();
    formData.append("file", file);

    setLoading(true);
    setResult(null);

    try {
      let response;
      if (predictionType === "disease") {
        response = await axios.post(
          `${API_BASE_URL}/predict/?prediction_type=disease`,
          formData,
          { headers: { "Content-Type": "multipart/form-data" } }
        );
      } else if (predictionType === "intercropping") {
        response = await axios.post(
          `${API_BASE_URL}/predict/intercropping/`,
          formData,
          { headers: { "Content-Type": "multipart/form-data" } }
        );
      }
      setResult(response.data);
    } catch (error) {
      setResult({ error: error.message, details: error.response?.data });
    }
    setLoading(false);
  };

  return (
    <div style={{ maxWidth: 600, margin: "2rem auto", fontFamily: "sans-serif" }}>
      <h2>Coconut Classifier Frontend</h2>
      <form onSubmit={handleSubmit}>
        <div>
          <label>
            Prediction Type:&nbsp;
            <select value={predictionType} onChange={e => setPredictionType(e.target.value)}>
              <option value="disease">Disease</option>
              <option value="intercropping">Intercropping</option>
            </select>
          </label>
        </div>
        <div>
          <input type="file" accept="image/*" onChange={handleFileChange} />
        </div>
        <button type="submit" disabled={loading}>
          {loading ? "Predicting..." : "Submit"}
        </button>
      </form>
      <hr />
      {result && (
        <div>
          <h3>Result:</h3>
          <pre style={{ background: "#f0f0f0", padding: "1em" }}>
            {JSON.stringify(result, null, 2)}
          </pre>
        </div>
      )}
    </div>
  );
}

export default App;
