import React, { useEffect, useState } from "react";
import axios from "axios";

function ShelfLifeTable() {
  const [records, setRecords] = useState([]);
  const [newRecord, setNewRecord] = useState({
    Ingredient: "",
    Shelf_Life_Days: "",
    Purchase_Rate: "",
    Unit: ""
  });

  const fetchData = async () => {
    try {
      const res = await axios.get("http://localhost:5000/shelf-life");
      setRecords(res.data);
    } catch (err) {
      console.error("Error fetching shelf-life records:", err);
    }
  };

  useEffect(() => {
    fetchData();
  }, []);

  const handleDelete = async (id) => {
    try {
      await axios.delete(`http://localhost:5000/shelf-life/${id}`);
      fetchData();
    } catch (err) {
      console.error("Delete error:", err);
    }
  };

  const handleChange = (e) => {
    setNewRecord({ ...newRecord, [e.target.name]: e.target.value });
  };

  const handleAdd = async () => {
    try {
      await axios.post("http://localhost:5000/shelf-life", newRecord);
      setNewRecord({ Ingredient: "", Shelf_Life_Days: "", Purchase_Rate: "", Unit: "" });
      fetchData();
    } catch (err) {
      console.error("Add error:", err);
    }
  };

  return (
    <div style={{ padding: "20px" }}>
      <h2>Shelf Life Collection</h2>
      <table border="1" cellPadding="8" style={{ width: "100%", marginTop: "10px" }}>
        <thead>
          <tr>
            <th>Ingredient</th>
            <th>Shelf Life (Days)</th>
            <th>Purchase Rate</th>
            <th>Unit</th>
            <th>Actions</th>
          </tr>
        </thead>
        <tbody>
          {records.map((item) => (
            <tr key={item._id}>
              <td>{item.Ingredient}</td>
              <td>{item.Shelf_Life_Days}</td>
              <td>{item.Purchase_Rate}</td>
              <td>{item.Unit}</td>
              <td>
                <button onClick={() => handleDelete(item._id)}>Delete</button>
              </td>
            </tr>
          ))}
          <tr>
            <td>
              <input
                type="text"
                name="Ingredient"
                value={newRecord.Ingredient}
                onChange={handleChange}
              />
            </td>
            <td>
              <input
                type="number"
                name="Shelf_Life_Days"
                value={newRecord.Shelf_Life_Days}
                onChange={handleChange}
              />
            </td>
            <td>
              <input
                type="number"
                step="0.01"
                name="Purchase_Rate"
                value={newRecord.Purchase_Rate}
                onChange={handleChange}
              />
            </td>
            <td>
              <input
                type="text"
                name="Unit"
                value={newRecord.Unit}
                onChange={handleChange}
              />
            </td>
            <td>
              <button onClick={handleAdd}>Add</button>
            </td>
          </tr>
        </tbody>
      </table>
    </div>
  );
}

export default ShelfLifeTable;
