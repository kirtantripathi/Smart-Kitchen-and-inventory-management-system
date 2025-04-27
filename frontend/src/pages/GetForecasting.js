import React, { useState } from "react";
import axios from "axios";

function GetForecasting() {
  const [inputValue, setInputValue] = useState("");
  const [message, setMessage] = useState("");
  const [forecastedData, setForecastedData] = useState(null);
  const [allocationData, setAllocationData] = useState(null);

  const handleChange = (e) => {
    setInputValue(e.target.value);
  };

  const handleSubmit = async () => {
    const partsArray = inputValue.split(",").map(item => item.trim());

    try {
      // Send the data to the backend for processing
      const response = await axios.post("http://localhost:5000/api/forecast-data", { data: partsArray });

      // Get the forecasted data and allocation data from the response
      setForecastedData(response.data.forecasted_data);
      setAllocationData(response.data.allocation_data);
      setMessage("✅ Forecast data retrieved successfully!");
      setInputValue(""); // Clear input
    } catch (err) {
      console.error(err);
      setMessage("❌ Failed to retrieve forecast data.");
    }
  };

  const renderTable = (title, data, columns) => {
    if (!data) return null;

    // Move "Ingredient" or "Date" to the top if necessary
    const orderedEntries = Object.entries(data).sort(([keyA], [keyB]) => {
      if (keyA === "Ingredient") return -1;
      if (keyB === "Ingredient") return 1;
      return 0;
    });

    return (
      <div style={{ flex: 1, margin: "10px", border: "1px solid #ccc", padding: "10px", borderRadius: "8px" }}>
        <h3>{title}</h3>
        <table className="table table-bordered" style={{ width: "100%", borderCollapse: "collapse" }}>
          <thead>
            <tr style={{ backgroundColor: "#f2f2f2" }}>
              {columns.map((column, index) => (
                <th key={index} style={{ padding: "8px", border: "1px solid #ddd", textAlign: "left" }}>
                  {column}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {data.map((item, index) => (
              <tr key={index}>
                {columns.map((column, colIndex) => (
                  <td key={colIndex} style={{ padding: "8px", border: "1px solid #ddd", textAlign: "left" }}>
                    {item[column] || "N/A"}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    );
  };

  return (
    <div style={{ maxWidth: "1000px", margin: "auto", padding: "20px" }}>
      <h2>GET Forecast Data</h2>
      <input
        type="text"
        value={inputValue}
        onChange={handleChange}
        placeholder="Enter Number of days for forecasting"
        style={{ width: "100%", padding: "8px", marginBottom: "10px" }}
      />
      <button onClick={handleSubmit} style={{ padding: "10px 20px" }}>
        Submit
      </button>

      {message && <p style={{ marginTop: "15px" }}>{message}</p>}

      {/* Container for the tables */}
      <div style={{ display: "flex", justifyContent: "space-between", marginTop: "20px" }}>
        
        {/* Render Forecasted Data Table */}
        {forecastedData && renderTable("Forecasted Results", forecastedData, ["Ingredient", "Total_Quantity_Required"])}

        {/* Render Allocation Data Table */}
        {allocationData && renderTable("Allocation Results", allocationData, ["Ingredient", "Allocated_Quantity", "Shelf_Life_Days"])}
      </div>
    </div>
  );
}

export default GetForecasting;
