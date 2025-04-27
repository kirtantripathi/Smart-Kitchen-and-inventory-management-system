// import React, { useEffect, useState } from 'react';
// import Papa from 'papaparse';
// import {
//   LineChart, Line, XAxis, YAxis,
//   CartesianGrid, Tooltip, ResponsiveContainer
// } from 'recharts';

// const csvFiles = [
//   'customer_sales_data_11_45.csv',
//   'daily_ingredient_consumption.csv',
//   'dish_costs.csv',
//   'ingredient_usage.csv',
//   'ingridient_detail.csv',
//   'mcd_sales_2_years.csv',
//   'purchase_table.csv',
//   'stock_table.csv'
// ];

// const Dashboard = () => {
//   const [data, setData] = useState({});
//   const [profitSummary, setProfitSummary] = useState(null);

//   useEffect(() => {
//     const loadCSVs = async () => {
//       const parsedData = {};

//       for (let file of csvFiles) {
//         const response = await fetch(`/data/${file}`); // ✅ fixed backtick issue
//         const text = await response.text();
//         const parsed = Papa.parse(text, { header: true, skipEmptyLines: true });
//         parsedData[file] = parsed.data;
//       }

//       setData(parsedData);
//       computeProfit(parsedData);
//     };

//     loadCSVs();
//   }, []);

//   const computeProfit = (parsedData) => {
//     const sales = parsedData['customer_sales_data_11_45.csv'] || [];
//     const purchases = parsedData['purchase_table.csv'] || [];
//     const wastage = parsedData['stock_table.csv'] || [];

//     let revenue = 0;
//     sales.forEach(s => {
//       revenue += parseFloat(s.total_amount || 0);
//     });

//     let ingredientCost = 0;
//     purchases.forEach(p => {
//       ingredientCost += parseFloat(p.total_price || 0);
//     });

//     let wastageCost = 0;
//     wastage.forEach(w => {
//       wastageCost += parseFloat(w.wastage_cost || 0);
//     });

//     const profit = revenue - (ingredientCost + wastageCost);

//     setProfitSummary({ revenue, ingredientCost, wastageCost, profit });
//   };

//   return (
//     <div className="p-6 space-y-6 bg-gray-100 min-h-screen">
//       <h1 className="text-3xl font-bold text-center text-green-700">📊 Loss-to-Profit Dashboard</h1>

//       {profitSummary && (
//         <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
//           {['Revenue', 'Ingredient Cost', 'Wastage Cost', 'Profit'].map((label, i) => (
//             <div key={i} className="bg-white p-4 rounded-xl shadow-lg text-center">
//               <p className="text-sm text-gray-500">{label}</p>
//               <h2 className={`text-xl font-bold ${label === 'Profit' ? 'text-green-600' : 'text-gray-800'}`}>
//                 ₹ {profitSummary[label.replace(' ', '').toLowerCase()].toFixed(2)}
//               </h2>
//             </div>
//           ))}
//         </div>
//       )}

//       {/* Revenue Trend Chart */}
//       {data['customer_sales_data_11_45.csv'] && (
//         <div className="bg-white p-4 rounded-xl shadow-xl">
//           <h2 className="text-lg font-semibold mb-2">Revenue Over Time</h2>
//           <ResponsiveContainer width="100%" height={300}>
//             <LineChart data={data['customer_sales_data_11_45.csv'].slice(0, 30)}>
//               <XAxis dataKey="date" />
//               <YAxis />
//               <Tooltip />
//               <CartesianGrid stroke="#eee" strokeDasharray="5 5" />
//               <Line type="monotone" dataKey="total_amount" stroke="#10b981" />
//             </LineChart>
//           </ResponsiveContainer>
//         </div>
//       )}
//     </div>
//   );
// };

// export default Dashboard;
