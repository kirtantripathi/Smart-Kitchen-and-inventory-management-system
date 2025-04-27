import React, { useState } from "react";
import axios from "axios";

function AddSalesData() {
  const [inputValue, setInputValue] = useState("");
  const [message, setMessage] = useState("");
  const [lastStockRow, setLastStockRow] = useState(null);
  const [lastConsumeRow, setLastConsumeRow] = useState(null);

  const handleChange = (e) => {
    setInputValue(e.target.value);
  };

  const handleSubmit = async () => {
    const partsArray = inputValue.split(",").map((item) => item.trim());

    try {
      const response = await axios.post("http://localhost:5000/api/save-data", {
        data: partsArray,
      });

      const { message, last_stock_row, last_consume_row } = response.data;

      setMessage("✅ " + message);
      setLastStockRow(last_stock_row);
      setLastConsumeRow(last_consume_row);
      setInputValue(""); // Clear input
    } catch (err) {
      console.error(err);
      setMessage("❌ Failed to save data.");
    }
  };

  const renderTable = (title, data) => {
    if (!data) return null;

    // Move Date to the top
    const orderedEntries = Object.entries(data).sort(([keyA], [keyB]) => {
      if (keyA === "Date") return -1;
      if (keyB === "Date") return 1;
      return 0;
    });

    return (
      <div style={{ flex: 1, margin: "10px", border: "1px solid #ccc", padding: "10px", borderRadius: "8px" }}>
        <h3>{title}</h3>
        <table border="1" cellPadding="6" style={{ width: "100%", borderCollapse: "collapse" }}>
          <thead>
            <tr style={{ backgroundColor: "#f2f2f2" }}>
              <th>Field</th>
              <th>Value</th>
            </tr>
          </thead>
          <tbody>
            {orderedEntries.map(([key, value], index) => (
              <tr key={index}>
                <td style={{ padding: "8px", border: "1px solid #ddd", textAlign: "left" }}>
                  <strong>{key}</strong>
                </td>
                <td style={{ padding: "8px", border: "1px solid #ddd", textAlign: "left" }}>
                  {value}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    );
  };

  return (
    <div style={{ maxWidth: "700px", margin: "auto", padding: "20px" }}>
      <h2>Add Sales Data</h2>
      <input
        type="text"
        value={inputValue}
        onChange={handleChange}
        placeholder="Enter comma separated values"
        style={{ width: "100%", padding: "8px", marginBottom: "10px" }}
      />
      <button onClick={handleSubmit} style={{ padding: "10px 20px" }}>
        Submit
      </button>
      {message && <p style={{ marginTop: "15px" }}>{message}</p>}

      {(lastStockRow || lastConsumeRow) && (
        <div style={{ display: "flex", justifyContent: "space-between", marginTop: "20px" }}>
          {renderTable("Last Stock Row", lastStockRow)}
          {renderTable("Last Consume Row", lastConsumeRow)}
        </div>
      )}
    </div>
  );
}

export default AddSalesData;
