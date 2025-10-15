import React from "react"
import axios from "axios"
import ReactDOM from "react-dom/client"
import { HelmetProvider } from "react-helmet-async"

import "./i18n"
import "./index.css"

const { protocol, hostname } = window.location

axios.defaults.baseURL = `${protocol}//${hostname}:8001`

import App from "./App.tsx"

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <HelmetProvider>
      <App />
    </HelmetProvider>
  </React.StrictMode>
)
