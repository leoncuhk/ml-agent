import React from 'react';
import { Modal, Button, message } from 'antd';
import { CopyOutlined } from '@ant-design/icons';
// Consider adding a syntax highlighter like react-syntax-highlighter
// import SyntaxHighlighter from 'react-syntax-highlighter';
// import { docco } from 'react-syntax-highlighter/dist/esm/styles/hljs';

interface CodeModalProps {
  isVisible: boolean;
  code: string;
  onClose: () => void;
}

const CodeModal: React.FC<CodeModalProps> = ({ isVisible, code, onClose }) => {

  const handleCopy = () => {
    navigator.clipboard.writeText(code)
      .then(() => {
        message.success('Code copied to clipboard!');
      })
      .catch(err => {
        message.error('Failed to copy code.');
        console.error('Clipboard copy failed:', err);
      });
  };

  return (
    <Modal
      title="Generated Python Code"
      open={isVisible} // Use 'open' instead of 'visible' for newer AntD versions
      onCancel={onClose}
      footer={[
        <Button key="copy" icon={<CopyOutlined />} onClick={handleCopy}>
          Copy Code
        </Button>,
        <Button key="close" onClick={onClose}>
          Close
        </Button>,
      ]}
      width={800} // Adjust width as needed
      styles={{ body: { maxHeight: '60vh', overflowY: 'auto' } }} // Use styles prop for body style
    >
      {/* Basic preformatted text display */}
      <pre style={{ background: '#f5f5f5', padding: '10px', borderRadius: '4px', whiteSpace: 'pre-wrap', wordWrap: 'break-word' }}>
        <code>{code}</code>
      </pre>
      {/* Example with SyntaxHighlighter (install it first: npm install react-syntax-highlighter @types/react-syntax-highlighter) */}
      {/*
      <SyntaxHighlighter language="python" style={docco} showLineNumbers>
        {code}
      </SyntaxHighlighter>
      */}
    </Modal>
  );
};

export default CodeModal;