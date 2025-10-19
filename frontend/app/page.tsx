"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import { Rocket, Download, List, CheckCircle, XCircle, Loader } from "lucide-react";

interface LogEntry {
  time: string;
  step: string;
  message: string;
}

interface KeyMoment {
  timestamp: number;
  description: string;
}

export default function Home() {
  const [reelUrl, setReelUrl] = useState("");
  const [isProcessing, setIsProcessing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [statusStep, setStatusStep] = useState("");
  const [statusMessage, setStatusMessage] = useState("");
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<{
    pdfId: string;
    pdfFilename: string;
    recipe: string;
    keyMoments: KeyMoment[];
  } | null>(null);
  const [debugEnabled, setDebugEnabled] = useState(false);

  const logContainerRef = useRef<HTMLDivElement>(null);

  const logDebug = useCallback((...args: unknown[]) => {
    if (debugEnabled) {
      console.log(...args);
    }
  }, [debugEnabled]);

  const logDebugError = useCallback((...args: unknown[]) => {
    if (debugEnabled) {
      console.error(...args);
    }
  }, [debugEnabled]);

  useEffect(() => {
    if (logContainerRef.current) {
      logContainerRef.current.scrollTop = logContainerRef.current.scrollHeight;
    }
  }, [logs]);

  useEffect(() => {
    logDebug("[DEBUG] Result state changed:", result);
    if (result) {
      logDebug("[DEBUG] Result is truthy, UI should display result section");
    } else {
      logDebug("[DEBUG] Result is null/undefined, UI will not display result section");
    }
  }, [result, logDebug]);

  const addLogEntry = (step: string, message: string) => {
    const time = new Date().toLocaleTimeString();
    setLogs((prevLogs) => [...prevLogs, { time, step, message }]);
  };

  const setErrorState = (message: string) => {
    setError(message);
    setStatusStep("ERROR");
    setStatusMessage(message);
    addLogEntry("ERROR", message);
  };

  const resetState = () => {
    setIsProcessing(true);
    setProgress(0);
    setStatusStep("INITIALIZING");
    setStatusMessage("Preparing to process...");
    setLogs([]);
    setError(null);
    setResult(null);
    addLogEntry("INIT", "Starting process...");
  };

  const handleProcessReel = () => {
    if (!reelUrl.trim()) {
      const message = "Please enter a reel URL.";
      setErrorState(message);
      return;
    }
    resetState();

    const serverUrl = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8080";
    const eventSource = new EventSource(
      `${serverUrl}/summarize?url=${encodeURIComponent(reelUrl)}`
    );

    eventSource.onmessage = (event) => {
      try {
        logDebug("[DEBUG] Raw SSE event received:", event.data);
        const data = JSON.parse(event.data);
        logDebug("[DEBUG] Parsed SSE data:", data);

        if (data.error) {
          logDebugError("[ERROR] Server sent error:", data.message);
          setErrorState(data.message);
          setIsProcessing(false);
          eventSource.close();
          return;
        }

        setProgress(data.progress || 0);
        setStatusStep(data.step.toUpperCase());
        setStatusMessage(data.message);
        addLogEntry(data.step.toUpperCase(), data.message);

        if (data.step === "complete") {
          logDebug("[DEBUG] Complete message received!");
          logDebug("[DEBUG] - pdf_id:", data.pdf_id || "MISSING");
          logDebug("[DEBUG] - pdf_filename:", data.pdf_filename || "MISSING");
          logDebug("[DEBUG] - recipe length:", data.recipe?.length || "MISSING");
          logDebug("[DEBUG] - key_moments count:", data.key_moments?.length || "MISSING");

          const resultData = {
            pdfId: data.pdf_id,
            pdfFilename: data.pdf_filename,
            recipe: data.recipe,
            keyMoments: data.key_moments,
          };

          logDebug("[DEBUG] Setting result state:", resultData);
          setResult(resultData);
          setIsProcessing(false);
          eventSource.close();
          logDebug("[DEBUG] Result state should now be set, UI should update");
        }
      } catch (e) {
        logDebugError("[ERROR] Error parsing SSE data:", e);
        logDebugError("[ERROR] Raw event data:", event.data);
        const errorMessage = "Failed to parse server event.";
        setErrorState(errorMessage);
        setIsProcessing(false);
        eventSource.close();
      }
    };

    eventSource.onerror = (err) => {
      logDebugError("EventSource failed:", err);
      const errorMessage = "Connection to server failed. Is it running?";
      setErrorState(errorMessage);
      setIsProcessing(false);
      eventSource.close();
    };
  };

  const downloadPDF = async () => {
    if (!result) return;
    const { pdfId, pdfFilename } = result;

    if (!pdfId) {
      const message = "PDF ID is missing. Cannot download the report.";
      setErrorState(message);
      logDebugError("Attempted to download PDF, but result.pdfId was missing.", result);
      return;
    }

    try {
      logDebug("[DEBUG] Downloading PDF with ID:", pdfId);
      const serverUrl = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8080";
      const downloadUrl = `${serverUrl}/download-pdf/${pdfId}?filename=${encodeURIComponent(pdfFilename)}`;

      logDebug("[DEBUG] Download URL:", downloadUrl);

      // Fetch the PDF from the server
      const response = await fetch(downloadUrl);

      if (!response.ok) {
        throw new Error(`Failed to download PDF: ${response.statusText}`);
      }

      // Convert response to blob
      const blob = await response.blob();
      logDebug("[DEBUG] PDF blob received, size:", blob.size, "bytes");

      // Create download link
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.style.display = "none";
      a.href = url;
      a.download = pdfFilename;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);

      logDebug("[DEBUG] PDF download initiated successfully");
    } catch (error) {
      logDebugError("[ERROR] Failed to download PDF:", error);
      const message = `Failed to download PDF: ${error instanceof Error ? error.message : error}`;
      setErrorState(message);
    }
  };

  return (
    <main className="flex min-h-screen flex-col items-center justify-center p-6">
      <div className="bg-white rounded-2xl shadow-2xl max-w-2xl w-full p-8 space-y-6">
        <div className="text-center">
          <h1 className="text-3xl font-bold text-gray-800">🎬 Reel Summarizer</h1>
          <p className="text-gray-500 mt-2">
            AI-powered video recipe extraction with real-time progress
          </p>
        </div>

        <div className="space-y-2">
          <label htmlFor="reelUrl" className="text-sm font-medium text-gray-600">
            Instagram Reel URL
          </label>
          <input
            type="text"
            id="reelUrl"
            value={reelUrl}
            onChange={(e) => setReelUrl(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && handleProcessReel()}
            placeholder="https://www.instagram.com/reel/..."
            className="w-full px-4 py-3 border-2 border-gray-200 rounded-lg focus:ring-2 focus:ring-purple-400 focus:border-purple-400 transition"
            disabled={isProcessing}
          />
        </div>

        <label className="flex items-center gap-2 text-sm text-gray-600">
          <input
            type="checkbox"
            className="h-4 w-4 rounded border-gray-300 text-purple-600 focus:ring-purple-500"
            checked={debugEnabled}
            onChange={(event) => setDebugEnabled(event.target.checked)}
          />
          Enable debug logging
        </label>

        <button
          onClick={handleProcessReel}
          disabled={isProcessing}
          className="w-full flex items-center justify-center gap-2 px-4 py-3 font-semibold text-white bg-gradient-to-r from-purple-500 to-indigo-600 rounded-lg shadow-md hover:shadow-lg hover:from-purple-600 hover:to-indigo-700 transition-all transform hover:-translate-y-0.5 disabled:bg-gray-400 disabled:from-gray-400 disabled:to-gray-500 disabled:cursor-not-allowed disabled:transform-none"
        >
          {isProcessing ? <><Loader className="animate-spin" /> Processing...</> : <><Rocket size={20} /> Process Reel</>}
        </button>

        {(isProcessing || error || result) && (
          <div className="pt-4 space-y-4">
            {/* Progress Bar and Status */}
            <div className="bg-gray-100 p-4 rounded-lg border-l-4 border-purple-500">
              <div className="flex justify-between items-center mb-2">
                <p className={`text-sm font-bold uppercase tracking-wider ${error ? 'text-red-500' : 'text-purple-600'}`}>
                  {statusStep}
                </p>
                <p className="text-sm font-semibold text-gray-700">{progress}%</p>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2.5 mb-2">
                <div
                  className="bg-gradient-to-r from-purple-500 to-indigo-600 h-2.5 rounded-full transition-all duration-300"
                  style={{ width: `${progress}%` }}
                ></div>
              </div>
              <p className={`text-sm ${error ? 'text-red-600' : 'text-gray-600'}`}>{statusMessage}</p>
            </div>

            {/* Log Viewer */}
            {debugEnabled && (
              <details className="group">
                <summary className="flex items-center gap-1 cursor-pointer text-purple-600 font-semibold">
                  <List size={16} /> View Detailed Log
                </summary>
                <div ref={logContainerRef} className="mt-2 bg-gray-50 border border-gray-200 rounded-lg p-3 max-h-48 overflow-y-auto text-xs font-mono text-gray-600 space-y-2">
                  {logs.map((log, i) => (
                    <div key={i} className="border-b border-gray-200 pb-1 last:border-b-0">
                      <span className="text-gray-400 mr-2">[{log.time}]</span>
                      <strong className="text-purple-700">{log.step}:</strong> {log.message}
                    </div>
                  ))}
                </div>
              </details>
            )}
          </div>
        )}

        {result && !error && (
          <div className="pt-4 space-y-4 animate-fade-in">
            <div className="text-center p-4 bg-green-50 border-l-4 border-green-500 rounded-lg">
              <h3 className="text-xl font-bold text-green-800 flex items-center justify-center gap-2">
                <CheckCircle /> Processing Complete!
              </h3>
            </div>

            <button
              onClick={downloadPDF}
              className="w-full flex items-center justify-center gap-2 px-4 py-3 font-semibold text-white bg-green-600 rounded-lg shadow-md hover:shadow-lg hover:bg-green-700 transition-all transform hover:-translate-y-0.5"
            >
              <Download size={20} /> Download PDF Report
            </button>

            <div className="space-y-2">
              <h4 className="font-bold text-gray-700">Recipe Preview:</h4>
              <pre className="bg-gray-50 border border-gray-200 rounded-lg p-4 text-sm whitespace-pre-wrap font-sans max-h-64 overflow-y-auto">
                {result.recipe}
              </pre>
            </div>
          </div>
        )}

        {error && !isProcessing && (
           <div className="pt-4 text-center p-4 bg-red-50 border-l-4 border-red-500 rounded-lg">
              <h3 className="text-xl font-bold text-red-800 flex items-center justify-center gap-2">
                <XCircle /> An Error Occurred
              </h3>
              <p className="text-red-700 mt-2">{error}</p>
            </div>
        )}
      </div>
    </main>
  );
}
