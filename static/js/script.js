document.getElementById("predict-btn").addEventListener("click", function() {
    const region = document.getElementById("region").value;
    const weatherInfo = document.getElementById("weather-info");
    const predictionInfo = document.getElementById("prediction-info");
    const weatherCard = document.getElementById("weather-card");
    const predictionCard = document.getElementById("prediction-card");

    weatherInfo.textContent = "Loading...";
    predictionInfo.textContent = "";
    weatherCard.style.opacity = "0.5";
    predictionCard.style.opacity = "0.5";

    console.log(`Sending request for region: ${region}`);

    fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ region: region })
    })
    .then(response => response.json())
    .then(data => {
        console.log("API Response:", data);
        if (data.error) {
            console.error("Error in API response:", data.error);
            weatherInfo.textContent = "Error: " + data.error;
            predictionInfo.textContent = "Prediction failed";
            weatherCard.style.opacity = "1";
            predictionCard.style.opacity = "1";
            return;
        }

        weatherInfo.innerHTML = `
            Region: ${data.region}<br>
            Rainfall: ${data.rainfall_prev} mm<br>
            Temp: ${data.temperature}°C<br>
            Humidity: ${data.humidity}%
        `;
        predictionInfo.innerHTML = `Next Hour: <span>${data.prediction} mm (${data.category || "Unknown"})</span>`;
        console.log(`Prediction: ${data.prediction} mm, Category: ${data.category || "Unknown"}`);
        weatherCard.style.opacity = "1";
        predictionCard.style.opacity = "1";
        weatherCard.style.transition = "opacity 0.5s";
        predictionCard.style.transition = "opacity 0.5s";

        // Style based on category
        predictionCard.className = "card";
        if (data.category) {
            predictionCard.classList.add(data.category.toLowerCase().replace(" ", "-") + "-rain");
            predictionInfo.style.color = data.category === "No Rain" ? "#00ff00" : "#e0e0e0";
        }

        // Alert for heavy rain
        if (data.category === "Heavy Rain") {
            alert(`Heavy Rain Alert for ${data.region}! Expected: ${data.prediction} mm. Stay safe!`);
        }
    })
    .catch(error => {
        console.error("Fetch error:", error);
        weatherInfo.textContent = "Error fetching data.";
        predictionInfo.textContent = "";
        weatherCard.style.opacity = "1";
        predictionCard.style.opacity = "1";
    });
});