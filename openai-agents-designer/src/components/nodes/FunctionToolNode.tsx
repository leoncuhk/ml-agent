import React, { memo, useCallback } from 'react';
import { Handle, Position, NodeProps, useReactFlow, Node } from 'reactflow';
import { Card, Typography, Input, Form, Select, Button } from 'antd';
import { PlusOutlined, DeleteOutlined } from '@ant-design/icons'; // Import icons

const { Text } = Typography;
const { TextArea } = Input;
const { Option } = Select;

// Define parameter structure
interface FunctionParameter {
  id: string;
  name: string;
  type: string;
}

const FunctionToolNode: React.FC<NodeProps> = ({ id, data }) => {
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

  const onNameChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    updateNodeData('name', event.target.value);
  };

   const onReturnTypeChange = (value: string) => {
    updateNodeData('returnType', value);
  };

  const onImplementationChange = (event: React.ChangeEvent<HTMLTextAreaElement>) => {
    updateNodeData('implementation', event.target.value);
  };

  // Parameter handling
  const handleParamChange = (paramId: string, field: 'name' | 'type', value: string) => {
    const updatedParams = (data.parameters || []).map((param: FunctionParameter) =>
      param.id === paramId ? { ...param, [field]: value } : param
    );
    updateNodeData('parameters', updatedParams);
  };

  const addParameter = () => {
    const newParam: FunctionParameter = { id: `param_${Date.now()}`, name: '', type: 'str' };
    const updatedParams = [...(data.parameters || []), newParam];
    updateNodeData('parameters', updatedParams);
  };

  const removeParameter = (paramId: string) => {
    const updatedParams = (data.parameters || []).filter((param: FunctionParameter) => param.id !== paramId);
    updateNodeData('parameters', updatedParams);
  };


  return (
    <Card
      size="small"
      title={<Text strong>Function Tool</Text>}
      style={{
        width: 350, // Adjusted width
        borderTop: '3px solid #f39c12', // Yellow top border
        background: '#fff',
      }}
      bodyStyle={{ padding: '12px' }}
    >
      {/* No Input Handle needed for Function Tool */}
      {/* Output Handle (Right - connects to Agent's tool input) */}
      <Handle type="source" position={Position.Right} id="a" style={{ top: '50%' }} />

      <Form layout="vertical" style={{ margin: 0 }}>
        <Form.Item label="Name:" style={{ marginBottom: 8 }}>
          <Input
            placeholder="Function Name (e.g., get_weather)"
            value={data.name || ''}
            onChange={onNameChange}
          />
        </Form.Item>

        <Form.Item label="Parameters:" style={{ marginBottom: 8 }}>
          {(data.parameters || []).map((param: FunctionParameter, index: number) => (
            <div key={param.id} style={{ display: 'flex', marginBottom: 4, gap: '4px' }}>
              <Input
                placeholder={`Param ${index + 1} Name`}
                value={param.name}
                onChange={(e) => handleParamChange(param.id, 'name', e.target.value)}
                style={{ flex: 1 }}
              />
              <Select
                value={param.type}
                onChange={(value) => handleParamChange(param.id, 'type', value)}
                style={{ width: 100 }}
              >
                <Option value="str">str</Option>
                <Option value="int">int</Option>
                <Option value="float">float</Option>
                <Option value="bool">bool</Option>
                {/* Add list, dict later if needed */}
              </Select>
              <Button icon={<DeleteOutlined />} onClick={() => removeParameter(param.id)} danger size="small" />
            </div>
          ))}
          <Button type="dashed" onClick={addParameter} block icon={<PlusOutlined />} size="small">
            Add Parameter
          </Button>
        </Form.Item>

        <Form.Item label="Return Type:" style={{ marginBottom: 8 }}>
           <Select
             placeholder="Select return type"
             value={data.returnType || 'str'}
             onChange={onReturnTypeChange}
           >
             <Option value="str">str</Option>
             <Option value="int">int</Option>
             <Option value="float">float</Option>
             <Option value="bool">bool</Option>
             <Option value="list">list</Option>
             <Option value="dict">dict</Option>
             <Option value="None">None</Option>
           </Select>
        </Form.Item>

        <Form.Item label="Implementation:" style={{ marginBottom: 8 }}>
          <TextArea
            rows={6}
            placeholder={`def ${data.name || 'function_name'}(...):\n  # Your Python code here\n  return ...`}
            value={data.implementation || ''}
            onChange={onImplementationChange}
            style={{ fontFamily: 'monospace' }} // Use monospace font for code
          />
        </Form.Item>
      </Form>
    </Card>
  );
};

export default memo(FunctionToolNode);