import React, { useCallback, useEffect } from 'react';
import { Node } from 'reactflow';
import { Form, Input, Typography, Switch } from 'antd';
import { Dispatch, SetStateAction } from 'react';

const { Text } = Typography;

interface RunnerPropertiesFormProps {
  node: Node;
  setNodes: Dispatch<SetStateAction<Node[]>>;
}

const RunnerPropertiesForm: React.FC<RunnerPropertiesFormProps> = ({ node, setNodes }) => {
  const [form] = Form.useForm();

  useEffect(() => {
    if (node) {
      form.setFieldsValue({
        input: node.data.input || '',
        executionMode: node.data.executionMode === 'async', // Convert to boolean for Switch
        context: node.data.context || '',
      });
    }
  }, [node, form]);

  const updateNodeData = useCallback((field: string, value: unknown) => {
    setNodes((nds: Node[]) =>
      nds.map((n) => {
        if (n.id === node.id) {
          const newData = { ...n.data, [field]: value };
          return { ...n, data: newData };
        }
        return n;
      })
    );
  }, [node.id, setNodes]);

  const handleValuesChange = (changedValues: Record<string, unknown>) => {
     for (const key in changedValues) {
       if (Object.prototype.hasOwnProperty.call(changedValues, key)) {
         let value = changedValues[key];
         // Convert boolean back to string for executionMode
         if (key === 'executionMode') {
           value = value ? 'async' : 'sync';
         }
         updateNodeData(key, value);
       }
     }
  };

  return (
    <Form
      form={form}
      layout="vertical"
      onValuesChange={handleValuesChange}
      key={node.id}
    >
      <Text strong style={{ marginBottom: '16px', display: 'block' }}>
        Edit Runner: {node.id}
      </Text>
      <Form.Item name="input" label="Input:">
        <Input placeholder="Enter initial input for the agent" />
      </Form.Item>
      <Form.Item name="executionMode" label="Execution Mode:" valuePropName="checked">
         <Switch checkedChildren="Async" unCheckedChildren="Sync" />
      </Form.Item>
      <Form.Item name="context" label="Context (Optional):">
        <Input placeholder="Optional context" />
      </Form.Item>
      {/* Display connected agent (read-only for now) */}
      <div style={{ marginBottom: 8 }}>
          <Text strong>Connected Agent:</Text>
          <div><Text type="secondary">{node.data.connectedAgent || '(Connect an Agent)'}</Text></div>
      </div>
    </Form>
  );
};

export default RunnerPropertiesForm;