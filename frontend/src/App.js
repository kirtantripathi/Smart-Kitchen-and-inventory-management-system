import React from "react";
import { BrowserRouter as Router, Route, Routes } from "react-router-dom";

import VegetableDetection from "./VegetableDetection";
import PacketDetection from "./PacketDetection";
import GetForecasting from "./pages/GetForecasting";
import ChefChat from "./components/ChefChatBot/ChefChat";
import WasteDetection from "./components/WasteDetection/WasteDetection";
import AddSalesData from "./AddSalesData";
import Navbar from "./components/Navbar";
import RecipeGenerator from "./components/RecipeGenerator/RecipeGenerator";
// import Dashboard from "./pages/Dashboard";

function App() {
  return (
    <Router>
      <div style={appContainerStyle}>
        <Navbar />
        <div style={pageContainerStyle}>
          <Routes>
            <Route path="/add-seles-data" element={<AddSalesData />} />
            <Route path="/vegetable-detection" element={<VegetableDetection />} />
            <Route path="/packet-detection" element={<PacketDetection />} />
            <Route path="/get-forecasting" element={<GetForecasting />} />
            <Route path="/chef-chat" element={<ChefChat />} />
            <Route path="/waste-detection" element={<WasteDetection />} />
            <Route path="/recipe-generator" element={<RecipeGenerator />} /> 
            {/* <Route path="/dashboard" element={<Dashboard />} /> */}
          </Routes>
        </div>
      </div>
    </Router>
  );
}

// 🖥 Styles
const appContainerStyle = {
  fontFamily: "'Segoe UI', sans-serif",
  backgroundColor: "#f0f4f8",
  minHeight: "100vh",
};

const pageContainerStyle = {
  maxWidth: "1200px",
  margin: "0 auto",
  padding: "30px 20px",
  backgroundColor: "#fff",
  borderRadius: "12px",
  boxShadow: "0 4px 12px rgba(0,0,0,0.1)",
};

export default App;