import React from "react";
import { Link, useLocation } from "react-router-dom";

const navItems = [
  // { path: "/dashboard", label: "Dashboard" },
  { path: "/add-seles-data", label: "Add Sales Data" },
  { path: "/vegetable-detection", label: "Health Detection" },
  { path: "/packet-detection", label: "Add to Inventory" },
  { path: "/get-forecasting", label: "Get Forecasting" },
  { path: "/chef-chat", label: "Chef Chat" },
  { path: "/waste-detection", label: "Waste Detection" },
  { path: "/recipe-generator", label: "Recipe Generator" },

];

function Navbar() {
  const location = useLocation();

  return (
    <nav style={navContainer}>
      <div style={brandStyle}>🥗 FreshServe</div>

      <ul style={navListStyle}>
        {navItems.map(({ path, label }) => (
          <li key={path} style={navItemStyle}>
            <Link
              to={path}
              style={{
                ...linkStyle,
                ...(location.pathname === path ? activeLinkStyle : {}),
              }}
            >
              {label}
              </Link>
          </li>
        ))}
      </ul>
    </nav>
  );
}

// 🎨 Clean, Balanced Styles
const navContainer = {
  background: "#f8f9fa",
  boxShadow: "0 2px 8px rgba(0,0,0,0.08)",
  borderRadius: "0 0 12px 12px",
  padding: "14px 32px",
  position: "sticky",
  top: 0,
  zIndex: 1000,
  display: "flex",
  justifyContent: "space-between",
  alignItems: "center",
};

const brandStyle = {
  fontSize: "20px",
  fontWeight: "600",
  color: "#495057",
  letterSpacing: "0.5px",
};

const navListStyle = {
  display: "flex",
  justifyContent: "flex-end",
  alignItems: "center",
  margin: 0,
  padding: 0,
  listStyle: "none",
  gap: "14px",
  flexWrap: "wrap",
};

const navItemStyle = {
  textAlign: "center",
};

const linkStyle = {
  color: "#333",
  textDecoration: "none",
  fontWeight: "500",
  padding: "10px 16px",
  fontSize: "15px",
  position: "relative",
  borderRadius: "8px",
  transition: "all 0.3s ease",
  display: "inline-block",
};

const activeLinkStyle = {
  color: "#ffffff",
  background: "#495057",
  transform: "scale(1.05)",
  fontWeight: "600",
};

// Add hover + underline style dynamically
const hoverStyle = `
  nav ul li a:hover {
    background-color: #e9ecef;
    color: #212529;
    border-radius: 8px;
  }
  nav ul li a .underline {
    position: absolute;
    left: 0;
    bottom: 6px;
    width: 0;
    height: 2px;
    background: #495057;
    transition: width 0.3s ease;
  }
 
`;
const style = document.createElement("style");
style.innerHTML = hoverStyle;
document.head.appendChild(style);

export default Navbar;