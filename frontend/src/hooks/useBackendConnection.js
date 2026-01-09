import { useState } from 'react';

export const useBackendConnection = (apiUrl) => {
  const [backendStatus, setBackendStatus] = useState("checking");
  const [statusMessage, setStatusMessage] = useState("Checking backend...");

  const checkBackendConnection = async () => {
    try {
      const res = await fetch(`${apiUrl}/health`);
      const data = await res.json();
      if (data.status === "healthy") {
        setBackendStatus("connected");
        setStatusMessage("Connected to backend");
      } else {
        setBackendStatus("error");
        setStatusMessage("Backend not ready");
      }
    } catch {
      setBackendStatus("error");
      setStatusMessage("Cannot connect to backend");
    }
  };

  return {
    backendStatus,
    statusMessage,
    checkBackendConnection
  };
};