import React, { useState } from "react";
import ShelfLifeTable from "../components/ShelfLife/ShelfLifeTable";
import IngredientUsageTable from "../components/IngredientUsage/IngredientUsageTable";
import CustomerPurchasesTable from "../components/CustomerPurchases/CustomerPurchasesTable";
import IngredientDatasetTable from "../components/IngredientDataset/IngredientDatasetTable";
import InventoryTable from "../components/InventoryData/InventoryTable"; // Import InventoryTable

function ManageCollections() {
  const [selected, setSelected] = useState("");

  const renderTable = () => {
    switch (selected) {
      case "Shelf Life":
        return <ShelfLifeTable />;
      case "Ingredient Usage":
        return <IngredientUsageTable />;
      
     
      case "Inventory":
        return <InventoryTable />; // Show InventoryTable when selected
      default:
        return <p>Select a collection to manage</p>;
    }
  };

  return (
    <div>
      <h2>Manage Collections</h2>
      <select onChange={(e) => setSelected(e.target.value)} value={selected}>
        <option value="">-- Select Collection --</option>
        <option value="Shelf Life">Shelf Life</option>
        <option value="Ingredient Usage">Ingredient Usage</option>
        
        
        <option value="Inventory">Inventory</option> {/* Add Inventory option */}
      </select>
      <div style={{ marginTop: "20px" }}>{renderTable()}</div>
    </div>
  );
}

export default ManageCollections;
