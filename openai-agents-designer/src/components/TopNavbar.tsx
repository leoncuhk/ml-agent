import React from 'react';
import { Button, Typography, Space } from 'antd'; // Import Typography and Space

const { Title } = Typography;

// Define props
interface TopNavbarProps {
  onGenerateCode: () => void;
}

const TopNavbar: React.FC<TopNavbarProps> = ({ onGenerateCode }) => { // Receive prop
  return (
    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', width: '100%' }}>
      <Title level={4} style={{ color: '#1677ff', margin: 0 }}>OpenAI Agents Workflow Designer</Title>
      <Space> {/* Use Space for button spacing */}
        {/* Add onClick handler */}
        <Button type="primary" onClick={onGenerateCode}>Generate Code</Button>
        {/* Optional Save/Load buttons */}
        {/* <Button>Save</Button> */}
        {/* <Button>Load</Button> */}
      </Space>
    </div>
  );
};

export default TopNavbar;