import React, { useState } from "react";
import { marked } from "marked";  // Import marked for Markdown parsing

const RecipeGenerator = () => {
  const [ingredients, setIngredients] = useState("");
  const [existingDishes, setExistingDishes] = useState("");
  const [recipes, setRecipes] = useState("");
  const [loading, setLoading] = useState(false);

  const handleGenerate = async () => {
    setLoading(true);
    setRecipes("");

    try {
      const response = await fetch("http://localhost:5000/generate-recipes", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          ingredients,
          existing_dishes: existingDishes
        })
      });

      const data = await response.json();
      if (data.recipes) {
        setRecipes(data.recipes); // Markdown content
      } else {
        setRecipes("Something went wrong!");
      }
    } catch (error) {
      setRecipes("Error connecting to backend!");
    }

    setLoading(false);
  };

  return (
    <div style={{ padding: "30px", fontFamily: "Arial" }}>
      <h2>🍳 AI Recipe Generator</h2>

      <div style={{ marginBottom: "15px" }}>
        <label>Available Ingredients:</label>
        <textarea
          rows="3"
          value={ingredients}
          onChange={(e) => setIngredients(e.target.value)}
          style={{ width: "100%", padding: "8px", marginTop: "5px" }}
          placeholder="e.g., paneer, bell peppers, garlic, tomato"
        />
      </div>

      <div style={{ marginBottom: "15px" }}>
        <label>Existing Dishes:</label>
        <textarea
          rows="2"
          value={existingDishes}
          onChange={(e) => setExistingDishes(e.target.value)}
          style={{ width: "100%", padding: "8px", marginTop: "5px" }}
          placeholder="e.g., Big Mac, Fries, McChicken"
        />
      </div>

      <button
        onClick={handleGenerate}
        style={{
          padding: "10px 20px",
          backgroundColor: "#007bff",
          color: "white",
          border: "none",
          cursor: "pointer",
          borderRadius: "5px"
        }}
        disabled={loading}
      >
        {loading ? "Generating..." : "Generate Recipes"}
      </button>

      <div style={{ marginTop: "30px" }}>
        <h3>Generated Recipes:</h3>
        <div
          style={{
            backgroundColor: "#f7f7f7",
            padding: "20px",
            borderRadius: "8px",
            border: "1px solid #ddd"
          }}
          dangerouslySetInnerHTML={{
            __html: marked.parse(recipes || "No recipes yet.")  // Convert Markdown to HTML
          }}
        />
      </div>
    </div>
  );
};

export default RecipeGenerator;
