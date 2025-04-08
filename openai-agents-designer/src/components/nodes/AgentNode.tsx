import React, { memo, useCallback } from 'react';
import { Handle, Position, NodeProps, useReactFlow, Node } from 'reactflow';
import { Card, Typography, Input, Form } from 'antd';

const { Text } = Typography;
const { TextArea } = Input;

// Restored version with internal form elements for direct editing (before PropertiesPanel integration)
// Note: This might conflict slightly with the PropertiesPanel approach later,
// but let's use this to ensure the node itself can render complex content.
const AgentNode: React.FC<NodeProps> = ({ id, data }) => {
  const { setNodes } = useReactFlow();

  const updateNodeData = useCallback((field: string, value: unknown) => {
    setNodes((nds: Node[]) =>
      nds.map((node) => {
        if (node.id === id) {
          const newData = { ...node.data, [field]: value };
          return { ...node, data: newData };
        }
        return node;
      })
    );
  }, [id, setNodes]);

  const onInputChange = (event: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>, field: string) => {
    updateNodeData(field, event.target.value);
  };

  return (
    <Card
      size="small"
      title={<Text strong>Agent</Text>}
      style={{
        width: 300,
        borderTop: '3px solid #3498db',
        background: '#fff',
      }}
      bodyStyle={{ padding: '12px' }}
    >
      <Handle type="target" position={Position.Left} id="a" style={{ top: '50%' }} />
      <Handle type="source" position={Position.Right} id="b" style={{ top: '50%' }} />

      <Form layout="vertical" style={{ margin: 0 }}>
        <Form.Item label="Name:" style={{ marginBottom: 8 }}>
          <Input
            placeholder="Agent Name"
            value={data.name || ''}
            onChange={(e) => onInputChange(e, 'name')}
          />
        </Form.Item>
        <Form.Item label="Instructions:" style={{ marginBottom: 8 }}>
          <TextArea
            rows={4}
            placeholder="Agent Instructions"
            value={data.instructions || ''}
            onChange={(e) => onInputChange(e, 'instructions')}
          />
        </Form.Item>
         {/* Read-only display for now */}
         <div style={{ marginBottom: 8 }}>
            <Text strong>Handoffs:</Text>
            <div><Text type="secondary">{data.handoffs?.join(', ') || '(None)'}</Text></div>
        </div>
         <div>
            <Text strong>Tools:</Text>
             <div><Text type="secondary">{data.tools?.join(', ') || '(None)'}</Text></div>
        </div>
      </Form>
    </Card>
  );
};

export default memo(AgentNode);