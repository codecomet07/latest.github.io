const apiUrl = "https://your-backend-app-name.onrender.com/predict";

async function classifyAudio() {
    const fileInput = document.getElementById("audio-upload");
    const resultText = document.getElementById("result-text");

    if (fileInput.files.length === 0) {
        resultText.textContent = "Please upload an audio file first.";
        resultText.style.color = "red";
        return;
    }

    const formData = new FormData();
    formData.append('audio', fileInput.files[0]);

    resultText.textContent = "Analyzing...";
    resultText.style.color = "blue";

    try {
        let response = await fetch(apiUrl, {
            method: "POST",
            body: formData
        });
        let result = await response.json();
        resultText.textContent = `Prediction: ${result.prediction}`;
        resultText.style.color = "green";
    } catch (error) {
        resultText.textContent = "Error in processing audio.";
        resultText.style.color = "red";
    }
}
