// Function to send comment to API & update frontend HTML w/ result
async function analyzeComment() {
    const commentBox = document.getElementById("userComment");
    if (!commentBox) return;

    const comment = commentBox.value;

    if (!comment) {
        alert("Please enter a comment before analyzing.");
        return;
    }

    try {
        const response = await fetch("/", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text: comment })
        });

        const data = await response.json();

        // Update result area
        const resultElem = document.getElementById("result");
        if (resultElem) {
            if (data.error) {
                resultElem.innerText = "Error: " + data.error;
            } else {
                resultElem.innerText =
                    `Label: ${data.label}, Confidence: ${(data.confidence * 100).toFixed(2)}%`;
            }
        }
    } catch (err) {
        console.error("Error calling API:", err);
        const resultElem = document.getElementById("result");
        if (resultElem) resultElem.innerText = "Error calling API.";
    }
}
