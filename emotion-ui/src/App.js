import React, { useState } from "react";

export default function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleFileChange = (event) => {
    setSelectedFile(event.target.files[0]);
    setResult(null);
  };

  const handleUpload = async () => {
    if (!selectedFile) return;
    setLoading(true);
    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      const response = await fetch("http://localhost:8000/predict", {
        method: "POST",
        body: formData,
      });
      const data = await response.json();
      setResult(data);
    } catch (error) {
      console.error("Upload failed:", error);
      setResult({ error: "Failed to analyze file." });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-r from-indigo-100 to-purple-100 flex items-center justify-center p-6">
      <div className="bg-white shadow-2xl rounded-2xl p-8 w-full max-w-xl text-center">
        <h1 className="text-3xl font-extrabold mb-6 text-indigo-700">🎵 Emotion Detector</h1>

        <input
          type="file"
          accept=".wav,.mp3"
          onChange={handleFileChange}
          className="mb-4 w-full border border-gray-300 rounded-md p-2 text-sm"
        />

        <button
          onClick={handleUpload}
          disabled={!selectedFile || loading}
          className="w-full bg-indigo-600 hover:bg-indigo-700 text-white font-bold py-2 px-4 rounded-lg disabled:bg-indigo-300"
        >
          {loading ? "Analyzing..." : "Upload and Analyze"}
        </button>

        {result && (
          <div className="mt-6 text-left bg-gray-50 border border-gray-200 rounded-lg p-5">
            {result.error ? (
              <p className="text-red-500 text-sm">{result.error}</p>
            ) : (
              <>
                <h2 className="font-semibold text-lg mb-2 text-gray-800">🎭 Main Emotion:</h2>
                <p className="text-xl font-bold text-indigo-600 mb-4">{result.emotion}</p>

                {result.probabilities && (
                  <>
                    <h3 className="font-medium text-sm mb-1 text-gray-700">All Emotions:</h3>
                    <ul className="text-sm list-disc list-inside text-gray-800">
                      {Object.entries(result.probabilities).map(([emotion, score]) => (
                        <li key={emotion}>
                          <span className="font-semibold">{emotion}</span>:{" "}
                          <span className="font-mono">{(score * 100).toFixed(2)}%</span>
                        </li>
                      ))}
                    </ul>
                  </>
                )}
              </>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
