const uploadUrl = "https://latest-github-io.onrender.com/upload-audio";
const predictUrl = "https://latest-github-io.onrender.com/predict";

async function classifyAudio() {
    const fileInput = document.getElementById("audio-upload");
    const resultText = document.getElementById("result-text");

    if (fileInput.files.length === 0) {
        resultText.textContent = "Please upload an audio file first.";
        resultText.style.color = "red";
        return;
    }

    const formData = new FormData();
    formData.append("file", fileInput.files[0]);  // Note: the Flask backend expects 'file'

    resultText.textContent = "Uploading...";
    resultText.style.color = "blue";

    try {
        // 1. Upload audio to Supabase via Flask
        const uploadResponse = await fetch(uploadUrl, {
            method: "POST",
            body: formData
        });

        const uploadResult = await uploadResponse.json();

        if (!uploadResult.url) {
            throw new Error("Upload failed.");
        }

        const audioUrl = uploadResult.url;
        resultText.textContent = "Analyzing...";
        resultText.style.color = "blue";

        // 2. Send the audio URL to your /predict endpoint
        const predictResponse = await fetch(predictUrl, {
            method: "POST",
            mode:"cors",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ audio_url: audioUrl })
        });

        const predictResult = await predictResponse.json();
        resultText.textContent = `Prediction: ${predictResult.prediction}`;
        resultText.style.color = "green";

    } catch (error) {
        console.error(error);
        resultText.textContent = "Error processing audio.";
        resultText.style.color = "red";
    }
}
