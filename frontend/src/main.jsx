import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App.jsx'
import {QueryClient,QueryClientProvider} from "react-query";
import './index.css'
import { PredictContextProvider } from './services/PredictionContextProvider';

const queryClient = new QueryClient();

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <QueryClientProvider client={queryClient}>
      <PredictContextProvider>
        <App />
      </PredictContextProvider>
    </QueryClientProvider>
  </React.StrictMode>,
)
