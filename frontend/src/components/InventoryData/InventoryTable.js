import React, { useState, useEffect } from "react";
import axios from "axios";

const InventoryTable = () => {
  const [inventory, setInventory] = useState([]);
  const [ingredient, setIngredient] = useState("");
  const [currentStock, setCurrentStock] = useState("");
  const [unit, setUnit] = useState("");
  const [lowStockThreshold, setLowStockThreshold] = useState("");
  const [reorderLevel, setReorderLevel] = useState("");
  const [expirationDate, setExpirationDate] = useState("");
  const [status, setStatus] = useState("");
  const [editingItem, setEditingItem] = useState(null); // To track the item being edited

  // Fetch inventory data from backend
  useEffect(() => {
    axios
      .get("http://localhost:5000/inventory") // Replace with your backend URL
      .then((response) => {
        setInventory(response.data);
      })
      .catch((error) => {
        console.error("Error fetching inventory data:", error);
      });
  }, []);

  // Auto-calculate the status based on conditions
  useEffect(() => {
    const calculateStatus = () => {
      const currentDate = new Date();
      const expiryDate = new Date(expirationDate);

      if (expiryDate < currentDate) {
        return "Expired";
      } else if (parseFloat(currentStock) <= parseInt(lowStockThreshold)) {
        return "Low Stock";
      } else {
        return "Available";
      }
    };

    if (currentStock && expirationDate) {
      const newStatus = calculateStatus();
      setStatus(newStatus);
    }
  }, [currentStock, expirationDate, lowStockThreshold]);

  // Handle form submission to add or update inventory
  const handleSubmit = (e) => {
    e.preventDefault();

    const newInventoryItem = {
      Ingredient: ingredient,
      Current_Stock: parseFloat(currentStock),
      Unit: unit,
      Low_Stock_Threshold: parseInt(lowStockThreshold),
      Reorder_Level: parseInt(reorderLevel),
      Expiration_Date: new Date(expirationDate),
      Status: status,
    };

    if (editingItem) {
      // If editing an existing item, update it
      axios
        .put(`http://localhost:5000/inventory/${editingItem._id}`, newInventoryItem)
        .then((response) => {
          const updatedInventory = inventory.map((item) =>
            item._id === editingItem._id ? response.data : item
          );
          setInventory(updatedInventory);
          alert("Inventory updated successfully!");
          resetForm();
        })
        .catch((error) => {
          console.error("Error updating inventory:", error);
        });
    } else {
      // If adding new item
      axios
        .post("http://localhost:5000/inventory", newInventoryItem)
        .then((response) => {
          setInventory([...inventory, response.data]);
          alert("Inventory added successfully!");
          resetForm();
        })
        .catch((error) => {
          console.error("Error adding inventory:", error);
        });
    }
  };

  // Handle delete inventory
  const handleDelete = (id) => {
    axios
      .delete(`http://localhost:5000/inventory/${id}`)
      .then(() => {
        setInventory(inventory.filter((item) => item._id !== id));
        alert("Inventory deleted successfully!");
      })
      .catch((error) => {
        console.error("Error deleting inventory:", error);
      });
  };

  // Handle editing an existing inventory item
  const handleEdit = (item) => {
    setIngredient(item.Ingredient);
    setCurrentStock(item.Current_Stock);
    setUnit(item.Unit);
    setLowStockThreshold(item.Low_Stock_Threshold);
    setReorderLevel(item.Reorder_Level);
    setExpirationDate(item.Expiration_Date);
    setStatus(item.Status);
    setEditingItem(item);
  };

  // Reset form fields
  const resetForm = () => {
    setIngredient("");
    setCurrentStock("");
    setUnit("");
    setLowStockThreshold("");
    setReorderLevel("");
    setExpirationDate("");
    setStatus("");
    setEditingItem(null);
  };

  return (
    <div>
      <h2>Inventory Management</h2>

      {/* Add new inventory or Edit existing inventory */}
      <form onSubmit={handleSubmit}>
        <div>
          <label>Ingredient: </label>
          <input
            type="text"
            value={ingredient}
            onChange={(e) => setIngredient(e.target.value)}
            required
          />
        </div>
        <div>
          <label>Current Stock: </label>
          <input
            type="number"
            value={currentStock}
            onChange={(e) => setCurrentStock(e.target.value)}
            required
          />
        </div>
        <div>
          <label>Unit: </label>
          <input
            type="text"
            value={unit}
            onChange={(e) => setUnit(e.target.value)}
            required
          />
        </div>
        <div>
          <label>Low Stock Threshold: </label>
          <input
            type="number"
            value={lowStockThreshold}
            onChange={(e) => setLowStockThreshold(e.target.value)}
            required
          />
        </div>
        <div>
          <label>Reorder Level: </label>
          <input
            type="number"
            value={reorderLevel}
            onChange={(e) => setReorderLevel(e.target.value)}
            required
          />
        </div>
        <div>
          <label>Expiration Date: </label>
          <input
            type="date"
            value={expirationDate}
            onChange={(e) => setExpirationDate(e.target.value)}
            required
          />
        </div>
        <div>
          <label>Status: </label>
          <input
            type="text"
            value={status}
            readOnly
            style={{ backgroundColor: "#f0f0f0" }} // Disable input to show calculated status
          />
        </div>
        <button type="submit">{editingItem ? "Update Inventory" : "Add Inventory"}</button>
      </form>

      {/* Inventory Table */}
      <table>
        <thead>
          <tr>
            <th>Ingredient</th>
            <th>Current Stock</th>
            <th>Unit</th>
            <th>Low Stock Threshold</th>
            <th>Reorder Level</th>
            <th>Expiration Date</th>
            <th>Status</th>
            <th>Actions</th>
          </tr>
        </thead>
        <tbody>
          {inventory.map((item) => (
            <tr key={item._id}>
              <td>{item.Ingredient}</td>
              <td>{item.Current_Stock}</td>
              <td>{item.Unit}</td>
              <td>{item.Low_Stock_Threshold}</td>
              <td>{item.Reorder_Level}</td>
              <td>{new Date(item.Expiration_Date).toLocaleDateString()}</td>
              <td>{item.Status}</td>
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

export default InventoryTable;
