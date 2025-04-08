import React, { useCallback, useEffect } from 'react';
import { Node } from 'reactflow';
import { Form, Input, Typography, Select, Button } from 'antd';
import { PlusOutlined, DeleteOutlined } from '@ant-design/icons';
import { Dispatch, SetStateAction } from 'react';

const { Text } = Typography;
const { TextArea } = Input;
const { Option } = Select;

// Define parameter structure (same as in FunctionToolNode)
interface FunctionParameter {
  id: string;
  name: string;
  type: string;
}

interface FunctionToolPropertiesFormProps {
  node: Node;
  setNodes: Dispatch<SetStateAction<Node[]>>;
}

const FunctionToolPropertiesForm: React.FC<FunctionToolPropertiesFormProps> = ({ node, setNodes }) => {
  const [form] = Form.useForm();

  useEffect(() => {
    if (node) {
      form.setFieldsValue({
        name: node.data.name || '',
        returnType: node.data.returnType || 'str',
        implementation: node.data.implementation || '',
        // Parameters are handled separately below
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
       if (Object.prototype.hasOwnProperty.call(changedValues, key) && key !== 'parameters') { // Exclude parameters
         updateNodeData(key, changedValues[key]);
       }
     }
  };

  // Parameter handling (similar to FunctionToolNode, but updates global state)
  const handleParamChange = (paramId: string, field: 'name' | 'type', value: string) => {
    const updatedParams = (node.data.parameters || []).map((param: FunctionParameter) =>
      param.id === paramId ? { ...param, [field]: value } : param
    );
    updateNodeData('parameters', updatedParams);
  };

  const addParameter = () => {
    const newParam: FunctionParameter = { id: `param_${Date.now()}`, name: '', type: 'str' };
    const updatedParams = [...(node.data.parameters || []), newParam];
    updateNodeData('parameters', updatedParams);
  };

  const removeParameter = (paramId: string) => {
    const updatedParams = (node.data.parameters || []).filter((param: FunctionParameter) => param.id !== paramId);
    updateNodeData('parameters', updatedParams);
  };


  return (
    <Form
      form={form}
      layout="vertical"
      onValuesChange={handleValuesChange}
      key={node.id}
    >
      <Text strong style={{ marginBottom: '16px', display: 'block' }}>
        Edit Function Tool: {node.data.name || node.id}
      </Text>
      <Form.Item name="name" label="Name:">
        <Input placeholder="Function Name (e.g., get_weather)" />
      </Form.Item>

      <Form.Item label="Parameters:">
        {(node.data.parameters || []).map((param: FunctionParameter, index: number) => (
          <div key={param.id} style={{ display: 'flex', marginBottom: 4, gap: '4px' }}>
            <Input
              placeholder={`Param ${index + 1} Name`}
              value={param.name} // Directly use value from node data
              onChange={(e) => handleParamChange(param.id, 'name', e.target.value)}
              style={{ flex: 1 }}
            />
            <Select
              value={param.type} // Directly use value from node data
              onChange={(value) => handleParamChange(param.id, 'type', value)}
              style={{ width: 100 }}
            >
              <Option value="str">str</Option>
              <Option value="int">int</Option>
              <Option value="float">float</Option>
              <Option value="bool">bool</Option>
            </Select>
            <Button icon={<DeleteOutlined />} onClick={() => removeParameter(param.id)} danger size="small" />
          </div>
        ))}
        <Button type="dashed" onClick={addParameter} block icon={<PlusOutlined />} size="small">
          Add Parameter
        </Button>
      </Form.Item>

      <Form.Item name="returnType" label="Return Type:">
         <Select placeholder="Select return type">
           <Option value="str">str</Option>
           <Option value="int">int</Option>
           <Option value="float">float</Option>
           <Option value="bool">bool</Option>
           <Option value="list">list</Option>
           <Option value="dict">dict</Option>
           <Option value="None">None</Option>
         </Select>
      </Form.Item>

      <Form.Item name="implementation" label="Implementation:">
        <TextArea
          rows={6}
          placeholder={`def ${node.data.name || 'function_name'}(...):\n  # Your Python code here\n  return ...`}
          style={{ fontFamily: 'monospace' }}
        />
      </Form.Item>
    </Form>
  );
};

export default FunctionToolPropertiesForm;