import React from 'react';
import { Card, Typography } from 'antd';
import { ApiOutlined, PlaySquareOutlined, ToolOutlined } from '@ant-design/icons'; // 示例图标

const { Title, Text } = Typography;

const DraggableNode: React.FC<{ type: string; label: string; icon: React.ReactNode; description: string }> = ({ type, label, icon, description }) => { // Add description prop
  const onDragStart = (event: React.DragEvent<HTMLDivElement>, nodeType: string) => {
    event.dataTransfer.setData('application/reactflow', nodeType);
    event.dataTransfer.effectAllowed = 'move';
  };

  return (
    <div
      onDragStart={(event) => onDragStart(event, type)}
      draggable
      style={{
        display: 'flex',
        alignItems: 'center',
        padding: '8px',
        marginBottom: '8px',
        border: '1px solid #d9d9d9',
        borderRadius: '4px',
        cursor: 'grab',
        background: '#f9f9f9', // Lighter background
        transition: 'background 0.2s ease',
      }}
      // Add hover effect (optional)
      // onMouseEnter={(e) => e.currentTarget.style.background = '#f0f0f0'}
      // onMouseLeave={(e) => e.currentTarget.style.background = '#f9f9f9'}
    >
      <span style={{ marginRight: '12px', fontSize: '16px' }}>{icon}</span> {/* Increased margin and icon size */}
      <div>
        <Text strong>{label}</Text>
        <div style={{ fontSize: '12px', color: '#888' }}>{description}</div> {/* Add description */}
      </div>
    </div>
  );
};


const SidePanel: React.FC = () => {
  return (
    <div style={{ padding: '16px' }}>
      <Title level={5} style={{ marginBottom: '16px' }}>Components</Title>
      <DraggableNode type="agent" label="Agent" icon={<ApiOutlined style={{ color: '#3498db' }} />} description="OpenAI Agent with instructions" />
      <DraggableNode type="runner" label="Runner" icon={<PlaySquareOutlined style={{ color: '#e74c3c' }} />} description="Executes an agent with input" />
      <DraggableNode type="functionTool" label="Function Tool" icon={<ToolOutlined style={{ color: '#f39c12' }} />} description="Custom function tool for agents" />
      {/* 可以添加 Guardrail Node 等 */}
      {/* <DraggableNode type="guardrail" label="Guardrail" icon={<SafetyOutlined style={{ color: '#9b59b6' }} />} /> */}

      <Card size="small" style={{ marginTop: '24px', background: '#f9f9f9' }}>
        <Text type="secondary">Drag components to the canvas to create a workflow. Connect nodes to establish relationships.</Text>
      </Card>

      {/* Add Connections Info Card */}
      <Title level={5} style={{ marginTop: '24px', marginBottom: '16px' }}>Connections:</Title>
      <Card size="small" style={{ background: '#f9f9f9' }}>
         <ul style={{ paddingLeft: '20px', margin: 0, fontSize: '12px', color: '#555' }}>
            <li>Agent → Agent: Handoff</li>
            <li>Function → Agent: Tool</li>
            <li>Agent → Runner: Execution</li>
         </ul>
      </Card>
    </div>
  );
};

export default SidePanel;