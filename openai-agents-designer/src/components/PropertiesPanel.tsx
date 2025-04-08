import React, { Dispatch, SetStateAction } from 'react'; // Import types
import { Node } from 'reactflow';
import { Typography, Empty } from 'antd';
// Import the specific form components
import AgentPropertiesForm from './AgentPropertiesForm';
import RunnerPropertiesForm from './RunnerPropertiesForm';
import FunctionToolPropertiesForm from './FunctionToolPropertiesForm';
// Removed duplicate commented import

const { Title } = Typography;

interface PropertiesPanelProps {
  selectedNode: Node | null;
  setNodes: Dispatch<SetStateAction<Node[]>>; // Add setNodes prop type
  // Removed duplicate setNodes declaration
}

const PropertiesPanel: React.FC<PropertiesPanelProps> = ({ selectedNode, setNodes }) => { // Receive setNodes

  const renderPropertiesForm = () => {
    if (!selectedNode) {
      return <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="Select a node to view its properties" />;
    }

    switch (selectedNode.type) {
      case 'agent':
        // Render AgentPropertiesForm, passing setNodes
        return <AgentPropertiesForm node={selectedNode} setNodes={setNodes} />;
      case 'runner':
         // Render RunnerPropertiesForm, passing setNodes
        return <RunnerPropertiesForm node={selectedNode} setNodes={setNodes} />;
      case 'functionTool':
         // Render FunctionToolPropertiesForm, passing setNodes
        return <FunctionToolPropertiesForm node={selectedNode} setNodes={setNodes} />;
      default:
        return <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description={`No properties editor for type: ${selectedNode.type}`} />;
    }
  };

  return (
    <div>
      <Title level={5} style={{ marginBottom: '16px' }}>Properties</Title>
      {renderPropertiesForm()}
    </div>
  );
};

export default PropertiesPanel;