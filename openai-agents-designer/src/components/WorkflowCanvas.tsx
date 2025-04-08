import React, { useCallback, useRef, Dispatch, SetStateAction } from 'react';
import ReactFlow, {
  Controls,
  Background,
  Node,
  Edge,
  Connection,
  BackgroundVariant,
  useReactFlow,
  NodeMouseHandler,
  OnNodesChange,
  OnEdgesChange,
} from 'reactflow';
import { nanoid } from 'nanoid';
import AgentNode from './nodes/AgentNode';
import RunnerNode from './nodes/RunnerNode';
import FunctionToolNode from './nodes/FunctionToolNode';

// Define custom node types
const nodeTypes = {
  agent: AgentNode,
  runner: RunnerNode,
  functionTool: FunctionToolNode,
};

// Helper function for generating unique IDs
const getId = () => `dndnode_${nanoid()}`;

// Define props for WorkflowCanvas including lifted state and handlers
interface WorkflowCanvasProps {
  nodes: Node[];
  edges: Edge[];
  onNodesChange: OnNodesChange;
  onEdgesChange: OnEdgesChange;
  onConnect: (connection: Connection) => void;
  setSelectedNode: Dispatch<SetStateAction<Node | null>>;
  setNodes: Dispatch<SetStateAction<Node[]>>;
}

// Receive all props from App
function WorkflowCanvas({
  nodes,
  edges,
  onNodesChange,
  onEdgesChange,
  onConnect,
  setSelectedNode,
  setNodes
}: WorkflowCanvasProps) {
  const reactFlowWrapper = useRef<HTMLDivElement>(null);
  const { project } = useReactFlow();

  // Drag and drop handlers
  const onDragOver = useCallback((event: React.DragEvent) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = 'move';
  }, []);

  const onDrop = useCallback(
    (event: React.DragEvent) => {
      event.preventDefault();

      if (!reactFlowWrapper.current) {
        return;
      }

      const type = event.dataTransfer.getData('application/reactflow');

      if (typeof type === 'undefined' || !type) {
        return;
      }

      const reactFlowBounds = reactFlowWrapper.current.getBoundingClientRect();
      const position = project({
        x: event.clientX - reactFlowBounds.left,
        y: event.clientY - reactFlowBounds.top,
      });

      let newNodeData = {};
      if (type === 'agent') {
        newNodeData = { name: 'New Agent', instructions: '' };
      } else if (type === 'runner') {
        newNodeData = { input: '', executionMode: 'sync' };
      } else if (type === 'functionTool') {
        newNodeData = { name: 'new_function', returnType: 'str', parameters: [] };
      } else {
         newNodeData = { label: `${type} node` }; // Fallback
      }

      const newNode: Node = {
        id: getId(),
        type,
        position,
        data: newNodeData,
      };

      setNodes((nds: Node[]) => nds.concat(newNode));
    },
    [project, setNodes]
  );

  // Function to validate connections
  const isValidConnection = useCallback(
    (connection: Connection): boolean => {
      const sourceNode = nodes.find((node) => node.id === connection.source);
      const targetNode = nodes.find((node) => node.id === connection.target);

      if (!sourceNode || !targetNode) {
        return false;
      }

      // Rule 1: FunctionTool ('a' source) -> Agent ('a' target)
      if (sourceNode.type === 'functionTool' && targetNode.type === 'agent') {
        return connection.targetHandle === 'a'; // FunctionTool only has source 'a' implicitly
      }
      // Rule 2: Agent ('b' source) -> Agent ('a' target)
      if (sourceNode.type === 'agent' && targetNode.type === 'agent') {
         return connection.sourceHandle === 'b' && connection.targetHandle === 'a';
      }
      // Rule 3: Agent ('b' source) -> Runner ('a' target)
       if (sourceNode.type === 'agent' && targetNode.type === 'runner') {
         return connection.sourceHandle === 'b' && connection.targetHandle === 'a';
      }
      // Rule 4: Prevent connecting Runner to anything other than Agent
      if (targetNode.type === 'runner' && sourceNode.type !== 'agent') {
          return false;
      }
      // Rule 5: Prevent connecting FunctionTool to anything other than Agent
      if (sourceNode.type === 'functionTool' && targetNode.type !== 'agent') {
          return false;
      }

      return false; // Disallow other connections
    },
    [nodes] // Depend on nodes state from props
  );

  // Handle node click
  const onNodeClick: NodeMouseHandler = useCallback((event, node) => {
    setSelectedNode(node);
  }, [setSelectedNode]);

   // Handle pane click
  const onPaneClick = useCallback(() => {
    setSelectedNode(null);
  }, [setSelectedNode]);


  return (
    <div style={{ height: '100%', width: '100%' }} ref={reactFlowWrapper} onDragOver={onDragOver} onDrop={onDrop}>
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        isValidConnection={isValidConnection}
        nodeTypes={nodeTypes}
        onNodeClick={onNodeClick}
        onPaneClick={onPaneClick}
        fitView
        defaultEdgeOptions={{ type: 'smoothstep' }}
      >
        <Controls />
        <Background variant={BackgroundVariant.Dots} gap={12} size={1} />
      </ReactFlow>
    </div>
  );
}

export default WorkflowCanvas;