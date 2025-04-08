import React, { useCallback, useEffect } from 'react';
import { Node } from 'reactflow';
import { Form, Input, Typography, Select } from 'antd';
import { Dispatch, SetStateAction } from 'react';

const { Text } = Typography;
const { TextArea } = Input;
const { Option } = Select;

interface AgentPropertiesFormProps {
  node: Node;
  setNodes: Dispatch<SetStateAction<Node[]>>;
}

const AgentPropertiesForm: React.FC<AgentPropertiesFormProps> = ({ node, setNodes }) => {
  const [form] = Form.useForm();

  // Update form fields when selected node changes
  useEffect(() => {
    if (node) {
      form.setFieldsValue({
        name: node.data.name || '',
        instructions: node.data.instructions || '',
        handoff_description: node.data.handoff_description || '',
        output_type: node.data.output_type || 'None',
      });
    }
  }, [node, form]);

  // Callback to update node data in the global state
  const updateNodeData = useCallback((field: string, value: unknown) => {
    setNodes((nds: Node[]) =>
      nds.map((n) => {
        if (n.id === node.id) {
          // Create a new data object with the updated field
          const newData = { ...n.data, [field]: value };
          // Return a new node object instance
          return { ...n, data: newData };
        }
        return n;
      })
    );
  }, [node.id, setNodes]);

  // Handle form value changes
  const handleValuesChange = (changedValues: Record<string, unknown>) => { // Changed any to Record<string, unknown>
     for (const key in changedValues) {
       if (Object.prototype.hasOwnProperty.call(changedValues, key)) {
         updateNodeData(key, changedValues[key]);
       }
     }
  };


  return (
    <Form
      form={form}
      layout="vertical"
      onValuesChange={handleValuesChange}
      // key={node.id} // Removed key for testing
    >
      <Text strong style={{ marginBottom: '16px', display: 'block' }}>
        Edit Agent: {node.data.name || node.id}
      </Text>
      <Form.Item name="name" label="Name:">
        <Input placeholder="Agent Name" />
      </Form.Item>
      <Form.Item name="instructions" label="Instructions:">
        <TextArea rows={5} placeholder="Agent Instructions" />
      </Form.Item>
      <Form.Item name="handoff_description" label="Handoff Description (Optional):">
        <Input placeholder="Describe when to handoff" />
      </Form.Item>
      <Form.Item name="output_type" label="Output Type:">
        <Select>
          <Option value="None">None</Option>
          <Option value="Custom" disabled>Custom Pydantic Model (Coming Soon)</Option>
        </Select>
      </Form.Item>
       {/* Display connected handoffs/tools (read-only for now) */}
       <div style={{ marginBottom: 8 }}>
            <Text strong>Handoffs:</Text>
            <div><Text type="secondary">{node.data.handoffs?.join(', ') || '(Connect other Agents)'}</Text></div>
        </div>
         <div>
            <Text strong>Tools:</Text>
             <div><Text type="secondary">{node.data.tools?.join(', ') || '(Connect Function Tools)'}</Text></div>
        </div>
    </Form>
  );
};

export default AgentPropertiesForm;