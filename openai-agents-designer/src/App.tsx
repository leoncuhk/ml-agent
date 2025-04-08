import React, { useState, useCallback } from 'react'; // Import useCallback
import { Layout, theme } from 'antd'; // Removed unused Modal import
import WorkflowCanvas from './components/WorkflowCanvas';
import TopNavbar from './components/TopNavbar';
import SidePanel from './components/SidePanel';
import PropertiesPanel from './components/PropertiesPanel';
// Import React Flow hooks and types needed for state management
import {
  Node,
  Edge,
  useNodesState,
  useEdgesState,
  addEdge,
  Connection,
} from 'reactflow';
// Import the code generation utility
import { generatePythonCode } from './utils/codeGenerator';
// import { generatePythonCode } from './utils/codeGenerator';
import CodeModal from './components/CodeModal'; // Import the modal component
import './App.css';

const { Header, Sider, Content } = Layout;

const App: React.FC = () => {
  const {
    token: { colorBgContainer },
  } = theme.useToken();

  // State for the selected node
  const [selectedNode, setSelectedNode] = useState<Node | null>(null);

  // Lifted state from WorkflowCanvas
  // Define initial state here or import if it gets large
  // Define initial state here or import if it gets large
  const initialNodes: Node[] = [
    { id: '1', type: 'agent', data: { name: 'Triage Agent', instructions: 'Determine which agent to use...' }, position: { x: 150, y: 25 } },
    { id: '2', type: 'runner', data: { input: 'Hola, ¿cómo estás?', executionMode: 'async' }, position: { x: 550, y: 50 } },
    { id: '3', type: 'functionTool', data: { name: 'get_weather', returnType: 'str', parameters: [{id: 'p1', name: 'city', type: 'str'}] }, position: { x: 150, y: 300 } }
  ];
  const initialEdges: Edge[] = [{ id: 'e1-2', source: '1', target: '2', type: 'smoothstep' }];

  const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges);

  // State for code generation modal
  const [isModalVisible, setIsModalVisible] = useState(false);
  const [generatedCode, setGeneratedCode] = useState('');

   // onConnect needs to be here now as it uses setEdges
   const onConnect = useCallback(
    (connection: Connection) => setEdges((eds) => addEdge({ ...connection, type: 'smoothstep' }, eds)),
    [setEdges],
  );

  // Function to handle code generation
  const handleGenerateCode = () => {
    // Use the actual code generation function
    const code = generatePythonCode(nodes, edges);
    setGeneratedCode(code);
    setIsModalVisible(true);
  };

  const handleCloseModal = () => {
    setIsModalVisible(false);
  };

  return (
    <Layout style={{ height: '100vh', overflow: 'hidden' /* Prevent outer scroll */ }}> {/* Use height: 100vh */}
      {/* Use TopNavbar component */}
      <Header style={{ display: 'flex', alignItems: 'center', padding: '0 16px', background: colorBgContainer, borderBottom: '1px solid #f0f0f0', zIndex: 1 /* Ensure header is above content */ }}>
        {/* Pass the handler to TopNavbar */}
        <TopNavbar onGenerateCode={handleGenerateCode} />
      </Header>
      {/* Ensure this inner Layout takes the remaining height */}
      {/* Explicitly set flex row direction */}
      <Layout style={{ display: 'flex', flexDirection: 'row', height: 'calc(100vh - 64px)', background: '#f0f2f5' /* Add a light background */ }}>
        {/* Use SidePanel component */}
        <Sider width={250} style={{ background: colorBgContainer, borderRight: '1px solid #f0f0f0', height: '100%', overflowY: 'auto' }}> {/* Ensure Sider takes full height of parent */}
          <SidePanel />
        </Sider>
        {/* Content should take the remaining space */}
        <Content
          style={{
            flex: 1,
            margin: 0,
            padding: 0,
            // background: colorBgContainer, // Remove background from content, let canvas handle it
            position: 'relative',
            height: '100%',
            overflow: 'hidden',
          }}
        >
          {/* Pass setSelectedNode to WorkflowCanvas */}
          <WorkflowCanvas
            nodes={nodes}
            edges={edges}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            onConnect={onConnect} // Pass onConnect
            setSelectedNode={setSelectedNode}
            setNodes={setNodes} // Pass setNodes for dropping new nodes
            // setEdges={setEdges} // Pass setEdges if needed inside canvas (optional for now)
          />
          {/* Removed dangling </ReactFlowProvider> tag */}
        </Content>
        {/* Right Properties Panel */}
        <Sider
          width={350} // Adjust width as needed
          style={{
            background: colorBgContainer,
            borderLeft: '1px solid #f0f0f0',
            height: '100%',
            overflowY: 'auto',
            padding: '16px' // Add padding
          }}
        >
          {/* Render PropertiesPanel, passing the selected node */}
          {/* Pass setNodes to PropertiesPanel for updates */}
          <PropertiesPanel selectedNode={selectedNode} setNodes={setNodes} />
        </Sider>
      </Layout>
      {/* Code Generation Modal - Moved inside the main Layout */}
      <CodeModal
        isVisible={isModalVisible}
        code={generatedCode}
        onClose={handleCloseModal}
      />
    </Layout>
  );
};

export default App;
