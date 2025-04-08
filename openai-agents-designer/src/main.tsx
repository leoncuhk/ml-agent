import React from 'react';
import ReactDOM from 'react-dom/client'
import App from './App.tsx'
import './index.css'
import 'reactflow/dist/style.css'; // 引入 React Flow 样式
import 'antd/dist/reset.css'; // 引入 Ant Design 重置样式

import { ReactFlowProvider } from 'reactflow'; // Import ReactFlowProvider here

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <ReactFlowProvider> {/* Wrap App with the provider */}
      <App />
    </ReactFlowProvider>
  </React.StrictMode>,
)
