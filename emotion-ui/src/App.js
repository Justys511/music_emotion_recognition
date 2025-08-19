import React, { useState } from "react";

// Базовый URL API:
// - в проде берётся из .env.production → REACT_APP_API_URL
// - локально (npm start) падаем на http://localhost:8000
const API_URL =
  process.env.REACT_APP_API_URL ||
  (process.env.NODE_ENV === "development" ? "http://localhost:8000" : "");

export default function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [errorText, setErrorText] = useState("");

  const handleFileChange = (event) => {
    setSelectedFile(event.target.files?.[0] || null);
    setResult(null);
    setErrorText("");
  };

  const handleUpload = async () => {
    if (!selectedFile) return;

    if (!API_URL) {
      setErrorText(
        "API_URL не задан. В проде задайте REACT_APP_API_URL в Vercel, локально используйте .env.local."
      );
      return;
    }

    setLoading(true);
    setErrorText("");

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const txt = await response.text().catch(() => "");
        throw new Error(`API ${response.status}: ${txt || "Request failed"}`);
      }

      const data = await response.json();
      setResult(data);
    } catch (error) {
      console.error("Upload failed:", error);
      setResult(null);
      setErrorText(error.message || "Failed to analyze file.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-r from-indigo-100 to-purple-100 flex items-center justify-center p-6">
      <div className="bg-white shadow-2xl rounded-2xl p-8 w-full max-w-xl text-center">
        <h1 className="text-3xl font-extrabold mb-6 text-indigo-700">
          🎵 Emotion Detector
        </h1>

        {/* подсказка какой API URL сейчас используется */}
        <p className="text-xs text-gray-500 mb-3">
          API: <span className="font-mono">{API_URL || "(не задан)"}</span>
        </p>

        <input
          type="file"
          accept=".wav,.mp3,audio/*"
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

        {errorText && (
          <div className="mt-4 text-left bg-red-50 border border-red-200 text-red-700 rounded-lg p-4 text-sm">
            {errorText}
          </div>
        )}

        {result && !errorText && (
          <div className="mt-6 text-left bg-gray-50 border border-gray-200 rounded-lg p-5">
            {result.error ? (
              <p className="text-red-500 text-sm">{result.error}</p>
            ) : (
              <>
                <h2 className="font-semibold text-lg mb-2 text-gray-800">
                  🎭 Main Emotion:
                </h2>
                <p className="text-xl font-bold text-indigo-600 mb-4">
                  {result.emotion}
                </p>

                {result.probabilities && (
                  <>
                    <h3 className="font-medium text-sm mb-1 text-gray-700">
                      All Emotions:
                    </h3>
                    <ul className="text-sm list-disc list-inside text-gray-800 space-y-1">
                      {Object.entries(result.probabilities).map(
                        ([emotion, score]) => (
                          <li key={emotion}>
                            <span className="font-semibold">{emotion}</span>:{" "}
                            <span className="font-mono">
                              {(Number(score) * 100).toFixed(2)}%
                            </span>
                          </li>
                        )
                      )}
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
