import React, { memo, useCallback } from 'react';
import { Handle, Position, NodeProps, useReactFlow, Node } from 'reactflow';
import { Card, Typography, Input, Form, Switch } from 'antd'; // Import Switch

const { Text } = Typography;

const RunnerNode: React.FC<NodeProps> = ({ id, data }) => {
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

  const onInputChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    updateNodeData('input', event.target.value);
  };

  const onModeChange = (checked: boolean) => {
    updateNodeData('executionMode', checked ? 'async' : 'sync');
  };

  return (
    <Card
      size="small"
      title={<Text strong>Runner</Text>}
      style={{
        width: 300,
        borderTop: '3px solid #e74c3c', // Red top border
        background: '#fff',
      }}
      bodyStyle={{ padding: '12px' }}
    >
      {/* Input Handle (Left - expects connection from Agent) */}
      <Handle type="target" position={Position.Left} id="a" style={{ top: '50%' }} />
      {/* Runner usually doesn't have an output handle in this context */}

      <Form layout="vertical" style={{ margin: 0 }}>
        <Form.Item label="Input:" style={{ marginBottom: 8 }}>
          <Input
            placeholder="Enter initial input for the agent"
            value={data.input || ''}
            onChange={onInputChange}
          />
        </Form.Item>
        <Form.Item label="Execution Mode:" style={{ marginBottom: 8 }}>
           <Switch
             checkedChildren="Async"
             unCheckedChildren="Sync"
             checked={data.executionMode === 'async'}
             onChange={onModeChange}
           />
        </Form.Item>
         {/* Placeholder for connected agent info */}
         <div style={{ marginBottom: 8 }}>
            <Text strong>Connected Agent:</Text>
             <div><Text type="secondary">{data.connectedAgent || '(Connect an Agent)'}</Text></div>
        </div>
        {/* Placeholder for context */}
        <Form.Item label="Context (Optional):" style={{ marginBottom: 8 }}>
          <Input placeholder="Optional context" disabled />
        </Form.Item>
      </Form>
    </Card>
  );
};

export default memo(RunnerNode);