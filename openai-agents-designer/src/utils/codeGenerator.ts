import { Node, Edge } from 'reactflow';

// Helper function to format parameters for function tools
// Define a more specific type for parameters if possible, or use unknown[]
interface FunctionParameter { id: string; name: string; type: string; } // Assuming structure from FunctionToolNode
const formatParameters = (parameters: FunctionParameter[] | undefined): string => {
  if (!parameters || parameters.length === 0) {
    return '';
  }
  return parameters.map(p => `${p.name}: ${p.type || 'str'}`).join(', ');
};

// Helper function to find connected nodes based on edge type and specific handles
const findConnectedNodes = (
    nodes: Node[],
    edges: Edge[],
    nodeId: string,
    handleType: 'source' | 'target',
    handleId?: string // Optional handle ID to filter connections
): Node[] => {
    return edges
        .filter(edge => {
            const isCorrectDirection = handleType === 'source' ? edge.source === nodeId : edge.target === nodeId;
            const isCorrectHandle = handleType === 'source'
                ? (!handleId || edge.sourceHandle === handleId)
                : (!handleId || edge.targetHandle === handleId);
            return isCorrectDirection && isCorrectHandle;
        })
        .map(edge => {
            const connectedNodeId = handleType === 'source' ? edge.target : edge.source;
            return nodes.find(node => node.id === connectedNodeId);
        })
        .filter((node): node is Node => node !== undefined); // Type guard to filter out undefined
};


export const generatePythonCode = (nodes: Node[], edges: Edge[]): string => {
  const imports = new Set<string>(['from agents import Agent, Runner']); // Use const
  const pydanticModels = ''; // Use const (will need to change if we add models later)
  let functionToolsCode = '';
  let agentDefinitions = '';
  let runnerCode = '';
  let mainFunctionContent = '';
  let needsAsyncio = false;

  const agentNodes = nodes.filter(node => node.type === 'agent');
  const runnerNodes = nodes.filter(node => node.type === 'runner');
  const functionToolNodes = nodes.filter(node => node.type === 'functionTool');

  // 1. Generate Function Tools
  if (functionToolNodes.length > 0) {
    imports.add('from agents import function_tool');
    functionToolNodes.forEach(node => {
      const funcName = node.data.name || `func_${node.id}`;
      const params = formatParameters(node.data.parameters);
      const returnType = node.data.returnType || 'str';
      const implementation = node.data.implementation || '  pass';
      functionToolsCode += `
@function_tool
def ${funcName}(${params}) -> ${returnType}:
  """${node.data.description || `Implementation for ${funcName}`}"""
${implementation.split('\n').map((line: string) => `  ${line}`).join('\n')} // Add type for line

`;
    });
  }

  // 2. Generate Agent Definitions
  agentNodes.forEach(node => {
    const agentVarName = (node.data.name || `agent_${node.id}`).toLowerCase().replace(/[^a-z0-9_]/g, '_');
    const agentName = node.data.name || `Agent ${node.id}`;
    const instructions = node.data.instructions || 'Default instructions.';
    const handoffDescription = node.data.handoff_description; // Get handoff description

    // Find connected tools (FunctionTool nodes connected to this agent's target handle 'a')
    const connectedToolNodes = findConnectedNodes(nodes, edges, node.id, 'target', 'a')
                                .filter(n => n.type === 'functionTool');
    const toolNames = connectedToolNodes
        .map(n => n.data.name || `func_${n.id}`)
        .filter(name => name);

     // Find connected handoffs (Agent nodes connected FROM this agent's source handle 'b')
     const connectedHandoffNodes = findConnectedNodes(nodes, edges, node.id, 'source', 'b')
                                    .filter(n => n.type === 'agent');
     const handoffVarNames = connectedHandoffNodes
        .map(n => (n.data.name || `agent_${n.id}`).toLowerCase().replace(/[^a-z0-9_]/g, '_'))
        .filter(name => name);


    let agentArgs = `name="${agentName}", instructions="${instructions}"`;
    if (toolNames.length > 0) {
      agentArgs += `, tools=[${toolNames.join(', ')}]`;
    }
    if (handoffDescription) {
        agentArgs += `, handoff_description="${handoffDescription}"`;
    }
    if (handoffVarNames.length > 0) {
        agentArgs += `, handoffs=[${handoffVarNames.join(', ')}]`;
    }
    // Add output_type later if needed
    // Add handoff_description later if needed
    // Add guardrails later if needed

    agentDefinitions += `${agentVarName} = Agent(${agentArgs})\n`;
  });


  // 3. Generate Runner Code
  runnerNodes.forEach(node => {
      const runnerVarName = `result_${node.id}`;
      const input = node.data.input || '';
      const executionMode = node.data.executionMode || 'sync';

      // Find the agent connected TO this runner's target handle 'a'
      const connectedAgentNodes = findConnectedNodes(nodes, edges, node.id, 'target', 'a')
                                    .filter(n => n.type === 'agent');

      let agentToRunVarName = 'None # Error: Runner not connected to an Agent';
      if (connectedAgentNodes.length > 0) {
          // Assuming only one agent can connect to a runner for now
          const sourceAgentNode = connectedAgentNodes[0];
          if (sourceAgentNode) {
              agentToRunVarName = (sourceAgentNode.data.name || `agent_${sourceAgentNode.id}`).toLowerCase().replace(/[^a-z0-9_]/g, '_');
          }
      }


      if (executionMode === 'async') {
          needsAsyncio = true;
          mainFunctionContent += `  ${runnerVarName} = await Runner.run(${agentToRunVarName}, input="${input}")\n`;
          mainFunctionContent += `  print(f"--- Output for Runner ${node.id} ---")\n`;
          mainFunctionContent += `  print(${runnerVarName}.final_output)\n\n`;
      } else {
          runnerCode += `${runnerVarName} = Runner.run_sync(${agentToRunVarName}, input="${input}")\n`;
          runnerCode += `print(f"--- Output for Runner ${node.id} ---")\n`;
          runnerCode += `print(${runnerVarName}.final_output)\n\n`;
      }
  });

  // 4. Assemble the final code
  let finalCode = Array.from(imports).join('\n') + '\n\n';
  if (needsAsyncio) {
      imports.add('import asyncio');
      finalCode = Array.from(imports).join('\n') + '\n\n'; // Re-join imports if asyncio was added
  }

  if (pydanticModels) {
    finalCode += `# Pydantic Models\n${pydanticModels}\n`;
  }
  if (functionToolsCode) {
    finalCode += `# Function Tools\n${functionToolsCode}\n`;
  }
  if (agentDefinitions) {
    finalCode += `# Agent Definitions\n${agentDefinitions}\n`;
  }

  if (needsAsyncio) {
      finalCode += `async def main():\n${mainFunctionContent}\n`;
      finalCode += `if __name__ == "__main__":\n    asyncio.run(main())\n`;
      if (runnerCode) { // Add sync runner code outside main if both exist
          finalCode += `\n# Synchronous Runner Calls\n${runnerCode}`;
      }
  } else {
      finalCode += `# Runner Calls\n${runnerCode}`;
  }


  return finalCode.trim();
};