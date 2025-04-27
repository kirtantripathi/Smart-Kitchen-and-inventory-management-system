import React, { useEffect, useState } from "react";
import axios from "axios";

const IngredientDatasetTable = () => {
  const [data, setData] = useState([]);
  const [newRecord, setNewRecord] = useState({
    Ingredient: "",
    Stock_Level: "",
    Consumption: "",
    Shelf_Life_Days: "",
    Purchase_Date: "",
    Expiration_Date: "",
    Date: ""
  });

  const [editingId, setEditingId] = useState(null);
  const [editFormData, setEditFormData] = useState({});

  const fetchData = async () => {
    const response = await axios.get("http://localhost:5000/api/ingredient-dataset");
    setData(response.data);
  };

  useEffect(() => {
    fetchData();
  }, []);

  const handleChange = (e, type = "new") => {
    const { name, value } = e.target;
    if (type === "new") {
      setNewRecord({ ...newRecord, [name]: value });
    } else {
      setEditFormData({ ...editFormData, [name]: value });
    }
  };

  const handleAdd = async () => {
    await axios.post("http://localhost:5000/api/ingredient-dataset", newRecord);
    setNewRecord({
      Ingredient: "",
      Stock_Level: "",
      Consumption: "",
      Shelf_Life_Days: "",
      Purchase_Date: "",
      Expiration_Date: "",
      Date: ""
    });
    fetchData();
  };

  const handleDelete = async (id) => {
    await axios.delete(`http://localhost:5000/api/ingredient-dataset/${id}`);
    fetchData();
  };

  const handleEditClick = (record) => {
    setEditingId(record._id);
    setEditFormData({ ...record });
  };

  const handleUpdate = async () => {
    await axios.put(`http://localhost:5000/api/ingredient-dataset/${editingId}`, editFormData);
    setEditingId(null);
    setEditFormData({});
    fetchData();
  };

  const formatDate = (dateStr) => dateStr ? new Date(dateStr).toISOString().split("T")[0] : "";

  return (
    <div>
      <h2>Ingredient Dataset Table</h2>
      <table border="1" cellPadding="10">
        <thead>
          <tr>
            <th>Ingredient</th>
            <th>Stock Level</th>
            <th>Consumption</th>
            <th>Shelf Life (Days)</th>
            <th>Purchase Date</th>
            <th>Expiration Date</th>
            <th>Added Date</th>
            <th>Actions</th>
          </tr>
        </thead>
        <tbody>
          {data.map((record) => (
            <tr key={record._id}>
              {editingId === record._id ? (
                <>
                  <td><input name="Ingredient" value={editFormData.Ingredient} onChange={(e) => handleChange(e, "edit")} /></td>
                  <td><input name="Stock_Level" value={editFormData.Stock_Level} onChange={(e) => handleChange(e, "edit")} /></td>
                  <td><input name="Consumption" value={editFormData.Consumption} onChange={(e) => handleChange(e, "edit")} /></td>
                  <td><input name="Shelf_Life_Days" value={editFormData.Shelf_Life_Days} onChange={(e) => handleChange(e, "edit")} /></td>
                  <td><input name="Purchase_Date" type="date" value={formatDate(editFormData.Purchase_Date)} onChange={(e) => handleChange(e, "edit")} /></td>
                  <td><input name="Expiration_Date" type="date" value={formatDate(editFormData.Expiration_Date)} onChange={(e) => handleChange(e, "edit")} /></td>
                  <td><input name="Date" type="date" value={formatDate(editFormData.Date)} onChange={(e) => handleChange(e, "edit")} /></td>
                  <td>
                    <button onClick={handleUpdate}>Update</button>
                  </td>
                </>
              ) : (
                <>
                  <td>{record.Ingredient}</td>
                  <td>{record.Stock_Level}</td>
                  <td>{record.Consumption}</td>
                  <td>{record.Shelf_Life_Days}</td>
                  <td>{formatDate(record.Purchase_Date)}</td>
                  <td>{formatDate(record.Expiration_Date)}</td>
                  <td>{formatDate(record.Date)}</td>
                  <td>
                    <button onClick={() => handleEditClick(record)}>Edit</button>
                    <button onClick={() => handleDelete(record._id)}>Delete</button>
                  </td>
                </>
              )}
            </tr>
          ))}
          <tr>
            <td><input name="Ingredient" value={newRecord.Ingredient} onChange={handleChange} /></td>
            <td><input name="Stock_Level" value={newRecord.Stock_Level} onChange={handleChange} /></td>
            <td><input name="Consumption" value={newRecord.Consumption} onChange={handleChange} /></td>
            <td><input name="Shelf_Life_Days" value={newRecord.Shelf_Life_Days} onChange={handleChange} /></td>
            <td><input name="Purchase_Date" type="date" value={newRecord.Purchase_Date} onChange={handleChange} /></td>
            <td><input name="Expiration_Date" type="date" value={newRecord.Expiration_Date} onChange={handleChange} /></td>
            <td><input name="Date" type="date" value={newRecord.Date} onChange={handleChange} /></td>
            <td><button onClick={handleAdd}>Add</button></td>
          </tr>
        </tbody>
      </table>
    </div>
  );
};

export default IngredientDatasetTable;
