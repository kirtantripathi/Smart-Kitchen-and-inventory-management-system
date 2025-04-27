import React, { useEffect, useState } from "react";
import axios from "axios";

const IngredientUsageTable = () => {
  const [usageData, setUsageData] = useState([]);
  const [form, setForm] = useState({ Dish: "", Ingredient: "", Quantity_per_Dish: "", Unit: "" });
  const [editId, setEditId] = useState(null);

  const fetchData = async () => {
    const res = await axios.get("http://localhost:5000/ingredient-usage");
    setUsageData(res.data);
  };

  useEffect(() => {
    fetchData();
  }, []);

  const handleChange = (e) => {
    setForm({ ...form, [e.target.name]: e.target.value });
  };

  const handleSubmit = async () => {
    if (editId) {
      await axios.put(`http://localhost:5000/ingredient-usage/${editId}`, form);
      setEditId(null);
    } else {
      await axios.post("http://localhost:5000/ingredient-usage", form);
    }
    setForm({ Dish: "", Ingredient: "", Quantity_per_Dish: "", Unit: "" });
    fetchData();
  };

  const handleEdit = (record) => {
    setEditId(record._id);
    setForm({
      Dish: record.Dish,
      Ingredient: record.Ingredient,
      Quantity_per_Dish: record.Quantity_per_Dish,
      Unit: record.Unit,
    });
  };

  const handleDelete = async (id) => {
    await axios.delete(`http://localhost:5000/ingredient-usage/${id}`);
    fetchData();
  };

  return (
    <div style={{ padding: "20px" }}>
      <h2>Ingredient Usage Management</h2>

      <div style={{ marginBottom: "20px" }}>
        <input type="text" name="Dish" placeholder="Dish" value={form.Dish} onChange={handleChange} />
        <input type="text" name="Ingredient" placeholder="Ingredient" value={form.Ingredient} onChange={handleChange} />
        <input type="number" name="Quantity_per_Dish" placeholder="Quantity" value={form.Quantity_per_Dish} onChange={handleChange} />
        <input type="text" name="Unit" placeholder="Unit" value={form.Unit} onChange={handleChange} />
        <button onClick={handleSubmit}>{editId ? "Update" : "Add"}</button>
      </div>

      <table border="1" cellPadding="10">
        <thead>
          <tr>
            <th>Dish</th>
            <th>Ingredient</th>
            <th>Quantity/Dish</th>
            <th>Unit</th>
            <th>Actions</th>
          </tr>
        </thead>
        <tbody>
          {usageData.map((item) => (
            <tr key={item._id}>
              <td>{item.Dish}</td>
              <td>{item.Ingredient}</td>
              <td>{item.Quantity_per_Dish}</td>
              <td>{item.Unit}</td>
              <td>
                <button onClick={() => handleEdit(item)}>Edit</button>
                <button onClick={() => handleDelete(item._id)}>Delete</button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

export default IngredientUsageTable;
