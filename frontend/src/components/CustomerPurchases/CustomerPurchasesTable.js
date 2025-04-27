import React, { useEffect, useState } from "react";
import axios from "axios";

const CustomerPurchasesTable = () => {
  const [data, setData] = useState([]);
  const [newRow, setNewRow] = useState({ Customer_ID: "", Date: "", Items_Purchased: "" });

  const fetchData = async () => {
    const res = await axios.get("http://localhost:5000/customer-purchases");
    setData(res.data);
  };

  const handleAdd = async () => {
    await axios.post("http://localhost:5000/customer-purchases", newRow);
    fetchData();
    setNewRow({ Customer_ID: "", Date: "", Items_Purchased: "" });
  };

  const handleDelete = async (id) => {
    await axios.delete(`http://localhost:5000/customer-purchases?_id=${id}`);
    fetchData();
  };

  const handleUpdate = async (id, field, value) => {
    const updatedRow = data.find((d) => d._id === id);
    updatedRow[field] = value;
    await axios.put("http://localhost:5000/customer-purchases", updatedRow);
    fetchData();
  };

  useEffect(() => {
    fetchData();
  }, []);

  return (
    <div>
      <h2 className="text-xl font-bold mb-4">Customer Purchases</h2>
      <table className="min-w-full bg-white border border-gray-300">
        <thead>
          <tr>
            <th className="border p-2">Customer ID</th>
            <th className="border p-2">Date</th>
            <th className="border p-2">Items Purchased</th>
            <th className="border p-2">Actions</th>
          </tr>
        </thead>
        <tbody>
          {data.map((row) => (
            <tr key={row._id}>
              <td className="border p-2">
                <input
                  value={row.Customer_ID}
                  onChange={(e) => handleUpdate(row._id, "Customer_ID", e.target.value)}
                />
              </td>
              <td className="border p-2">
                <input
                  type="date"
                  value={row.Date?.substring(0, 10)}
                  onChange={(e) => handleUpdate(row._id, "Date", e.target.value)}
                />
              </td>
              <td className="border p-2">
                <input
                  value={row.Items_Purchased}
                  onChange={(e) => handleUpdate(row._id, "Items_Purchased", e.target.value)}
                />
              </td>
              <td className="border p-2">
                <button
                  className="bg-red-500 text-white px-2 py-1 rounded"
                  onClick={() => handleDelete(row._id)}
                >
                  Delete
                </button>
              </td>
            </tr>
          ))}

          {/* Add New Row */}
          <tr>
            <td className="border p-2">
              <input
                value={newRow.Customer_ID}
                onChange={(e) => setNewRow({ ...newRow, Customer_ID: e.target.value })}
              />
            </td>
            <td className="border p-2">
              <input
                type="date"
                value={newRow.Date}
                onChange={(e) => setNewRow({ ...newRow, Date: e.target.value })}
              />
            </td>
            <td className="border p-2">
              <input
                value={newRow.Items_Purchased}
                onChange={(e) => setNewRow({ ...newRow, Items_Purchased: e.target.value })}
              />
            </td>
            <td className="border p-2">
              <button
                className="bg-green-500 text-white px-2 py-1 rounded"
                onClick={handleAdd}
              >
                Add
              </button>
            </td>
          </tr>
        </tbody>
      </table>
    </div>
  );
};

export default CustomerPurchasesTable;
